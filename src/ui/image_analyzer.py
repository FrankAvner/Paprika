import json
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median

import cv2

from PySide6.QtCore import (
    Qt,
    QObject,
    QThread,
    Signal,
    QTimer,
)

from PySide6.QtGui import QPixmap

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QProgressBar,
    QTextEdit,
    QSizePolicy,
)


UI_DIR = Path(__file__).resolve().parent

ANALYSIS_DIR = UI_DIR.parent / "analysis"
DATABASE_DIR = UI_DIR.parent / "database"

if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

if str(DATABASE_DIR) not in sys.path:
    sys.path.insert(0, str(DATABASE_DIR))

if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))


from leaf_segmentation.leaf_segmenter import LeafSegmenter, SegmentationStopped
from roi_selector import ROISelector
from database_service import DatabaseService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "images" / "sam2.1_b.pt"

RESULTS_ROOT = PROJECT_ROOT / "results" / "segmentation"

PERFORMANCE_HISTORY_FILE = RESULTS_ROOT / "performance_history.json"

INITIAL_BASELINE_SECONDS = 169.67
INITIAL_BASELINE_PIXELS = 546 * 572


class PerformanceHistory:

    def __init__(self, history_file):
        self.history_file = Path(history_file)

    def load(self):
        if not self.history_file.exists():
            return []

        try:
            with self.history_file.open(
                "r",
                encoding="utf-8"
            ) as file:
                data = json.load(file)

            if isinstance(data, list):
                return data

        except Exception as exc:
            print(
                "[PERFORMANCE HISTORY] "
                f"Could not load history: {exc}",
                flush=True
            )

        return []

    def save_run(
        self,
        model_name,
        extension,
        width,
        height,
        duration_seconds
    ):
        self.history_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        history = self.load()

        entry = {
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "model": model_name,
            "extension": extension,
            "width": width,
            "height": height,
            "megapixels": (width * height) / 1_000_000,
            "duration_seconds": round(
                duration_seconds,
                3
            ),
        }

        history.append(entry)

        with self.history_file.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                history,
                file,
                indent=4,
                ensure_ascii=False
            )

    def estimate_duration(
        self,
        model_name,
        extension,
        width,
        height
    ):
        megapixels = (width * height) / 1_000_000

        history = self.load()
        relevant = []

        for entry in history:
            if entry.get("model") != model_name:
                continue

            if entry.get("extension") != extension:
                continue

            entry_mp = entry.get("megapixels")
            entry_duration = entry.get("duration_seconds")

            if not entry_mp:
                continue

            if not entry_duration:
                continue

            normalized_time = entry_duration / entry_mp
            relevant.append(normalized_time)

        if relevant:
            seconds_per_mp = median(relevant)
            estimated = seconds_per_mp * megapixels
            return max(estimated, 1.0)

        baseline_mp = INITIAL_BASELINE_PIXELS / 1_000_000
        baseline_seconds_per_mp = INITIAL_BASELINE_SECONDS / baseline_mp
        estimated = baseline_seconds_per_mp * megapixels

        return max(estimated, 1.0)


class SegmentationWorker(QObject):

    stage = Signal(str)
    progress = Signal(int)
    busy = Signal(bool)
    finished = Signal(object)
    error = Signal(str, str)
    cancelled = Signal()

    def __init__(
        self,
        image_path,
        model_path,
        roi_rect=None
    ):
        super().__init__()

        self.image_path = Path(image_path)
        self.model_path = Path(model_path)
        self.roi_rect = roi_rect
        self.stop_requested = False
        self.segmenter = None

        self.media_id = None
        self.run_id = None
        self.database_service = DatabaseService()
        self.worker_start_time = None

    def log(self, message):
        print(
            f"[SEGMENTATION WORKER] {message}",
            flush=True
        )

    def request_stop(self):
        self.stop_requested = True
        self.log("STOP requested by user.")
        if self.segmenter is not None:
            self.segmenter.request_stop()

    def update_database_status(self, status):
        if self.run_id is None:
            return

        if self.worker_start_time is not None:
            duration = (
                datetime.now() - self.worker_start_time
            ).total_seconds()
        else:
            duration = 0

        try:
            self.database_service.update_run(
                self.run_id,
                status=status,
                duration_seconds=duration
            )
        except Exception as exc:
            print(
                "[DATABASE] "
                f"Failed to update run status to {status}: {exc}",
                flush=True
            )

    def check_stop(self):
        if self.stop_requested:
            self.update_database_status("stopped")

            self.busy.emit(False)
            self.stage.emit("SEGMENTATION STOPPED BY USER")
            self.log("Segmentation stopped.")
            self.cancelled.emit()

            return True

        return False

    def create_run_directory(self):
        RESULTS_ROOT.mkdir(
            parents=True,
            exist_ok=True
        )

        base_name = self.image_path.stem

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        run_name = f"{base_name}_{timestamp}"

        run_directory = RESULTS_ROOT / run_name

        run_directory.mkdir(
            parents=True,
            exist_ok=False
        )

        for directory_name in (
            "original",
            "masks",
            "leaves",
            "overlay",
            "roi",
        ):
            (
                run_directory / directory_name
            ).mkdir(
                parents=True,
                exist_ok=True
            )

        destination = (
            run_directory
            / "original"
            / self.image_path.name
        )

        import shutil

        shutil.copy2(
            self.image_path,
            destination
        )

        self.log(
            f"Run directory created: {run_directory}"
        )

        return run_directory

    def prepare_input(self, image, run_directory):
        height, width = image.shape[:2]

        if self.roi_rect is None:
            return (
                image,
                0,
                0,
                width,
                height,
                None
            )

        x1, y1, x2, y2 = self.roi_rect

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1, min(x2, width - 1))
        y2 = max(y1, min(y2, height - 1))

        roi = image[y1:y2 + 1, x1:x2 + 1]

        if roi.size == 0:
            raise ValueError("Selected ROI is empty.")

        roi_path = run_directory / "roi" / "roi.png"

        if not cv2.imwrite(str(roi_path), roi):
            raise IOError(
                f"Failed to save ROI:\n{roi_path}"
            )

        self.log("ROI selected.")
        self.log(
            f"ROI coordinates: ({x1},{y1}) - ({x2},{y2})"
        )
        self.log(
            f"ROI size: {x2 - x1 + 1} x {y2 - y1 + 1}"
        )

        return (
            roi,
            x1,
            y1,
            x2 - x1 + 1,
            y2 - y1 + 1,
            roi_path
        )

    def create_database_records(
        self,
        full_width,
        full_height
    ):
        self.worker_start_time = datetime.now()

        now = datetime.now()
        file_stat = self.image_path.stat()

        file_created_at = datetime.fromtimestamp(
            file_stat.st_ctime
        ).isoformat(timespec="seconds")

        file_modified_at = datetime.fromtimestamp(
            file_stat.st_mtime
        ).isoformat(timespec="seconds")

        capture_date = now.strftime("%Y-%m-%d")
        capture_time = now.strftime("%H:%M:%S")

        self.media_id = self.database_service.database.create_media(
            file_name=self.image_path.name,
            file_path=str(self.image_path),
            file_type="image",
            extension=self.image_path.suffix.lower(),
            width=full_width,
            height=full_height,
            capture_date=capture_date,
            capture_time=capture_time,
            file_created_at=file_created_at,
            file_modified_at=file_modified_at,
        )

        self.run_id = self.database_service.create_run(
            run_name=self.run_name,
            media_id=self.media_id,
            source_file=self.image_path.name,
            source_path=str(self.image_path),
            capture_date=capture_date,
            capture_time=capture_time,
            image_width=full_width,
            image_height=full_height,
            roi=self.roi_rect,
        )

        self.database_service.update_run(
            self.run_id,
            model_name=self.model_path.name,
            model_path=str(self.model_path),
            status="running"
        )

        self.log(
            "Database records created: "
            f"media_id={self.media_id}, "
            f"run_id={self.run_id}"
        )

    def convert_leaf_to_original(
        self,
        leaf,
        offset_x,
        offset_y,
        full_width,
        full_height
    ):
        roi_mask = leaf["mask"]

        roi_height, roi_width = roi_mask.shape[:2]

        import numpy as np

        full_mask = np.zeros(
            (full_height, full_width),
            dtype=roi_mask.dtype
        )

        paste_height = min(
            roi_height,
            full_height - offset_y
        )

        paste_width = min(
            roi_width,
            full_width - offset_x
        )

        if paste_height <= 0 or paste_width <= 0:
            return None

        full_mask[
            offset_y:offset_y + paste_height,
            offset_x:offset_x + paste_width
        ] = roi_mask[
            :paste_height,
            :paste_width
        ]

        x1, y1, x2, y2 = leaf["bbox"]

        x1 += offset_x
        y1 += offset_y
        x2 += offset_x
        y2 += offset_y

        x1 = max(0, min(x1, full_width - 1))
        y1 = max(0, min(y1, full_height - 1))
        x2 = max(x1, min(x2, full_width - 1))
        y2 = max(y1, min(y2, full_height - 1))

        area = int((full_mask > 0).sum())

        converted = dict(leaf)
        converted["mask"] = full_mask
        converted["bbox"] = (x1, y1, x2, y2)
        converted["area"] = area
        converted["width"] = x2 - x1 + 1
        converted["height"] = y2 - y1 + 1
        converted["center_x"] = (x1 + x2) / 2.0
        converted["center_y"] = (y1 + y2) / 2.0

        return converted

    def run(self):
        try:
            self.stage.emit(
                "STEP 1/8 - Checking input image..."
            )

            image = cv2.imread(str(self.image_path))

            if image is None:
                raise ValueError(
                    f"Could not read image:\n{self.image_path}"
                )

            full_height, full_width = image.shape[:2]

            self.log(
                f"Image size: {full_width} x {full_height}"
            )

            self.progress.emit(5)

            if self.check_stop():
                return

            self.stage.emit(
                "STEP 2/8 - Checking SAM 2 model..."
            )

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"SAM 2 model not found:\n{self.model_path}"
                )

            self.progress.emit(10)

            if self.check_stop():
                return

            self.stage.emit(
                "STEP 3/8 - Creating result directory..."
            )

            run_directory = self.create_run_directory()
            self.run_name = run_directory.name

            self.progress.emit(15)

            if self.check_stop():
                return

            self.stage.emit(
                "STEP 4/8 - Preparing analysis region..."
            )

            (
                analysis_image,
                offset_x,
                offset_y,
                analysis_width,
                analysis_height,
                roi_path
            ) = self.prepare_input(
                image,
                run_directory
            )

            if self.roi_rect is not None:
                self.stage.emit(
                    "ROI selected - "
                    f"{analysis_width} x {analysis_height}"
                )
            else:
                self.stage.emit("Full image selected.")

            self.create_database_records(
                full_width,
                full_height
            )

            self.progress.emit(20)

            if self.check_stop():
                return

            self.stage.emit(
                "STEP 5/8 - Loading SAM 2 model..."
            )

            self.segmenter = LeafSegmenter(
                model_path=str(self.model_path),
                tile_size=1024,
                tile_overlap=0.35,
                min_area_ratio=0.00002,
                max_area_ratio=0.30,
                duplicate_iou=0.82,
                containment=0.94,
                imgsz=1024,
                prompt_stride=72,
                max_prompt_points=80,
                split_min_area=700,
                split_min_seed_distance=24,
                sam_conf=0.10,
                max_tiles=16,
                max_total_prompt_points=240,
            )

            self.segmenter.load_model()

            self.progress.emit(25)

            if self.check_stop():
                return

            self.stage.emit(
                "STEP 6/8 - Running SAM 2 segmentation..."
            )

            self.busy.emit(True)

            if self.roi_rect is None:
                segmentation_input_path = self.image_path
            else:
                segmentation_input_path = roi_path

            leaves = self.segmenter.segment(
                str(segmentation_input_path)
            )

            self.busy.emit(False)

            if self.check_stop():
                return

            converted_leaves = []

            for leaf in leaves:
                converted = self.convert_leaf_to_original(
                    leaf,
                    offset_x,
                    offset_y,
                    full_width,
                    full_height
                )

                if converted is not None:
                    converted_leaves.append(converted)

            leaves = converted_leaves

            total_masks = len(leaves)

            self.log(
                "Inference completed. "
                f"Masks detected: {total_masks}"
            )

            self.progress.emit(65)

            self.stage.emit(
                "Segmentation completed - "
                f"{total_masks} masks"
            )

            self.stage.emit(
                "STEP 7/8 - Processing masks and leaves..."
            )

            overlay = image.copy()

            for index, leaf in enumerate(leaves):
                if self.check_stop():
                    return

                leaf_id = leaf["id"]

                message = (
                    f"Processing mask {index + 1}/{total_masks}"
                )

                self.stage.emit(message)
                self.log(message)

                mask = leaf["mask"]

                colored_mask = (image * 0).astype(image.dtype)

                colored_mask[mask > 0] = (0, 255, 0)

                overlay = cv2.addWeighted(
                    overlay,
                    0.70,
                    colored_mask,
                    0.30,
                    0
                )

                mask_path = (
                    run_directory
                    / "masks"
                    / f"mask_{leaf_id:03d}.png"
                )

                if not cv2.imwrite(str(mask_path), mask):
                    raise IOError(
                        f"Failed to save mask:\n{mask_path}"
                    )

                x1, y1, x2, y2 = leaf["bbox"]

                crop = image[y1:y2 + 1, x1:x2 + 1]

                if crop.size > 0:
                    leaf_path = (
                        run_directory
                        / "leaves"
                        / f"leaf_{leaf_id:03d}.png"
                    )

                    if not cv2.imwrite(
                        str(leaf_path),
                        crop
                    ):
                        raise IOError(
                            f"Failed to save leaf:\n{leaf_path}"
                        )

                progress = (
                    65
                    + int(
                        (
                            (index + 1)
                            / max(total_masks, 1)
                        )
                        * 20
                    )
                )

                self.progress.emit(progress)

            if self.check_stop():
                return

            self.stage.emit(
                "STEP 8/8 - Saving overlay and finalizing..."
            )

            overlay_path = (
                run_directory
                / "overlay"
                / "overlay.jpg"
            )

            if not cv2.imwrite(
                str(overlay_path),
                overlay
            ):
                raise IOError(
                    f"Failed to save overlay:\n{overlay_path}"
                )

            self.update_database_status("completed")

            self.progress.emit(100)

            completed_message = (
                f"COMPLETED - {total_masks} masks"
            )

            self.stage.emit(completed_message)

            self.finished.emit(
                {
                    "mask_count": total_masks,
                    "run_directory": str(run_directory),
                    "overlay": str(overlay_path),
                    "width": analysis_width,
                    "height": analysis_height,
                    "full_width": full_width,
                    "full_height": full_height,
                    "roi": self.roi_rect,
                    "media_id": self.media_id,
                    "run_id": self.run_id,
                }
            )

        except SegmentationStopped:
            self.busy.emit(False)
            self.update_database_status("stopped")
            self.stage.emit("SEGMENTATION STOPPED BY USER")
            self.cancelled.emit()

        except Exception as exc:
            self.busy.emit(False)

            self.update_database_status("failed")

            error_type = type(exc).__name__
            error_message = str(exc)
            full_traceback = traceback.format_exc()

            print("", flush=True)
            print("=" * 80, flush=True)
            print("LEAF SEGMENTATION ERROR", flush=True)
            print(f"ERROR TYPE: {error_type}", flush=True)
            print(f"ERROR MESSAGE: {error_message}", flush=True)
            print("-" * 80, flush=True)
            print(full_traceback, flush=True)
            print("=" * 80, flush=True)

            self.error.emit(
                error_type,
                f"ERROR MESSAGE:\n{error_message}\n\n"
                f"FULL TRACEBACK:\n{full_traceback}"
            )


class ImageAnalysis(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "PAPRIKA - IMAGE ANALYSIS"
        )

        self.resize(1000, 700)

        self.selected_image = None
        self.original_image_width = None
        self.original_image_height = None
        self.roi_rect = None

        self.thread = None
        self.worker = None
        self.roi_dialog = None

        self.run_start_time = None
        self.estimated_total_seconds = None

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(
            self.update_runtime_display
        )

        self.performance_history = PerformanceHistory(
            PERFORMANCE_HISTORY_FILE
        )

        self.create_ui()

    def create_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("IMAGE ANALYSIS")
        title.setStyleSheet(
            "font-size: 28px; font-weight: bold;"
        )
        layout.addWidget(title)

        image_group = QGroupBox("INPUT IMAGE")
        image_layout = QHBoxLayout(image_group)

        self.select_button = QPushButton("SELECT IMAGE")
        self.image_name_label = QLabel("No image selected")

        image_layout.addWidget(self.select_button)
        image_layout.addWidget(self.image_name_label)

        layout.addWidget(image_group)

        preview_group = QGroupBox("IMAGE PREVIEW")
        preview_layout = QVBoxLayout(preview_group)

        self.image_preview = QLabel("Select an image")
        self.image_preview.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.image_preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_preview.setMinimumSize(1, 1)
        self.image_preview.setStyleSheet(
            "border: 1px solid gray;"
        )

        preview_layout.addWidget(self.image_preview)
        layout.addWidget(preview_group)

        region_layout = QHBoxLayout()

        self.partial_region_button = QPushButton(
            "PARTIAL REGION"
        )

        self.clear_region_button = QPushButton(
            "CLEAR REGION"
        )

        self.region_label = QLabel(
            "REGION: FULL IMAGE"
        )

        self.partial_region_button.setEnabled(False)
        self.clear_region_button.setEnabled(False)

        region_layout.addWidget(
            self.partial_region_button
        )
        region_layout.addWidget(
            self.clear_region_button
        )
        region_layout.addWidget(self.region_label)

        layout.addLayout(region_layout)

        segmentation_group = QGroupBox(
            "LEAF SEGMENTATION"
        )

        segmentation_layout = QVBoxLayout(
            segmentation_group
        )

        button_layout = QHBoxLayout()

        self.segment_button = QPushButton(
            "RUN LEAF SEGMENTATION"
        )

        self.segment_button.setMinimumHeight(45)
        self.segment_button.setEnabled(False)

        self.stop_button = QPushButton(
            "STOP SEGMENTATION"
        )

        self.stop_button.setMinimumHeight(45)
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.segment_button)
        button_layout.addWidget(self.stop_button)

        segmentation_layout.addLayout(button_layout)

        current_label = QLabel("CURRENT OPERATION:")
        current_label.setStyleSheet(
            "font-weight: bold;"
        )
        segmentation_layout.addWidget(current_label)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(
            "font-weight: bold;"
        )
        segmentation_layout.addWidget(self.status_label)

        timing_group = QGroupBox("PROCESSING TIME")
        timing_layout = QVBoxLayout(timing_group)

        self.start_time_label = QLabel(
            "START TIME: -"
        )
        self.elapsed_time_label = QLabel(
            "ELAPSED: 00:00"
        )
        self.remaining_time_label = QLabel(
            "ESTIMATED REMAINING: -"
        )
        self.end_time_label = QLabel(
            "ESTIMATED END: -"
        )
        self.estimate_basis_label = QLabel(
            "ESTIMATE BASIS: -"
        )

        timing_layout.addWidget(self.start_time_label)
        timing_layout.addWidget(self.elapsed_time_label)
        timing_layout.addWidget(self.remaining_time_label)
        timing_layout.addWidget(self.end_time_label)
        timing_layout.addWidget(self.estimate_basis_label)

        segmentation_layout.addWidget(timing_group)

        log_label = QLabel("ACTIVITY LOG:")
        log_label.setStyleSheet(
            "font-weight: bold;"
        )
        segmentation_layout.addWidget(log_label)

        self.stage_log = QTextEdit()
        self.stage_log.setReadOnly(True)
        self.stage_log.setMinimumHeight(60)
        self.stage_log.setMaximumHeight(150)

        segmentation_layout.addWidget(self.stage_log)

        progress_label = QLabel("PROGRESS:")
        progress_label.setStyleSheet(
            "font-weight: bold;"
        )
        segmentation_layout.addWidget(progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        segmentation_layout.addWidget(self.progress_bar)
        layout.addWidget(segmentation_group)

        self.select_button.clicked.connect(
            self.select_image
        )
        self.partial_region_button.clicked.connect(
            self.open_partial_region
        )
        self.clear_region_button.clicked.connect(
            self.clear_region
        )
        self.segment_button.clicked.connect(
            self.run_segmentation
        )
        self.stop_button.clicked.connect(
            self.stop_segmentation
        )

    def log(self, message):
        print(
            f"[IMAGE ANALYSIS] {message}",
            flush=True
        )

    def add_stage_message(self, message):
        timestamp = datetime.now().strftime(
            "%H:%M:%S"
        )

        self.stage_log.append(
            f"[{timestamp}] {message}"
        )

        scrollbar = self.stage_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def format_duration(self, seconds):
        seconds = max(0, int(seconds))

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        if hours > 0:
            return (
                f"{hours:02d}:"
                f"{minutes:02d}:"
                f"{secs:02d}"
            )

        return (
            f"{minutes:02d}:"
            f"{secs:02d}"
        )

    def select_image(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Image",
                r"C:\paprika\images",
                (
                    "Image Files "
                    "(*.jpg *.jpeg *.png *.bmp "
                    "*.tif *.tiff *.webp)"
                )
            )

            if not file_path:
                return

            self.selected_image = Path(file_path)

            image = cv2.imread(
                str(self.selected_image)
            )

            if image is None:
                raise ValueError(
                    f"Could not read image:\n"
                    f"{self.selected_image}"
                )

            self.original_image_height, (
                self.original_image_width
            ) = image.shape[:2]

            self.image_name_label.setText(
                self.selected_image.name
            )

            self.roi_rect = None

            self.image_preview.setPixmap(
                QPixmap(str(self.selected_image))
            )

            self.update_image_preview()

            self.partial_region_button.setEnabled(True)
            self.clear_region_button.setEnabled(False)
            self.segment_button.setEnabled(True)
            self.stop_button.setEnabled(False)

            self.status_label.setText(
                "Image selected - ready"
            )

            self.region_label.setText(
                "REGION: FULL IMAGE"
            )

            self.stage_log.clear()

            self.add_stage_message(
                "Image selected successfully."
            )

            self.add_stage_message(
                "Choose PARTIAL REGION or "
                "run segmentation on the full image."
            )

            self.start_time_label.setText(
                "START TIME: -"
            )
            self.elapsed_time_label.setText(
                "ELAPSED: 00:00"
            )
            self.remaining_time_label.setText(
                "ESTIMATED REMAINING: -"
            )
            self.end_time_label.setText(
                "ESTIMATED END: -"
            )
            self.estimate_basis_label.setText(
                "ESTIMATE BASIS: -"
            )

            self.progress_bar.setMaximum(100)
            self.progress_bar.setValue(0)

        except Exception as exc:
            self.handle_error(
                "IMAGE SELECTION ERROR",
                exc
            )

    def open_partial_region(self):
        if self.selected_image is None:
            QMessageBox.warning(
                self,
                "PARTIAL REGION",
                "Please select an image first."
            )
            return

        if self.thread is not None:
            return

        try:
            self.status_label.setText(
                "Waiting for region selection..."
            )

            self.add_stage_message(
                "Opening PARTIAL REGION selection window."
            )

            self.roi_dialog = ROISelector(
                str(self.selected_image),
                self
            )

            self.roi_dialog.confirmed.connect(
                self.on_roi_confirmed
            )

            self.roi_dialog.cancelled.connect(
                self.on_roi_cancelled
            )

            self.roi_dialog.exec()

        except Exception as exc:
            self.roi_dialog = None
            self.handle_error(
                "PARTIAL REGION ERROR",
                exc
            )

    def on_roi_confirmed(self, roi):
        self.roi_rect = tuple(roi)

        x1, y1, x2, y2 = self.roi_rect

        roi_width = x2 - x1 + 1
        roi_height = y2 - y1 + 1

        self.region_label.setText(
            "REGION: "
            f"X1={x1} "
            f"Y1={y1} "
            f"X2={x2} "
            f"Y2={y2} "
            f"| SIZE={roi_width}x{roi_height}"
        )

        self.clear_region_button.setEnabled(True)

        self.add_stage_message(
            "ROI confirmed: "
            f"({x1},{y1}) - ({x2},{y2})"
        )

        self.status_label.setText(
            "ROI confirmed - starting segmentation..."
        )

        self.roi_dialog = None

        self.run_segmentation()

    def on_roi_cancelled(self):
        self.roi_dialog = None

        self.status_label.setText(
            "ROI selection cancelled."
        )

        self.add_stage_message(
            "ROI selection cancelled."
        )

    def clear_region(self):
        self.roi_rect = None

        self.clear_region_button.setEnabled(False)

        self.region_label.setText(
            "REGION: FULL IMAGE"
        )

        self.status_label.setText(
            "Full image selected"
        )

        self.add_stage_message(
            "ROI cleared - full image will be analyzed."
        )

    def update_image_preview(self):
        if self.selected_image is None:
            return

        pixmap = QPixmap(str(self.selected_image))
        if pixmap.isNull():
            return

        target_size = self.image_preview.contentsRect().size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            return

        scaled_pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_preview.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.selected_image is not None:
            self.update_image_preview()

    def run_segmentation(self):
        if self.selected_image is None:
            QMessageBox.warning(
                self,
                "IMAGE ANALYSIS",
                "Please select an image first."
            )
            return

        if self.thread is not None:
            return

        image = cv2.imread(
            str(self.selected_image)
        )

        if image is None:
            QMessageBox.critical(
                self,
                "IMAGE ANALYSIS",
                "Could not read the selected image."
            )
            return

        height, width = image.shape[:2]

        if self.roi_rect is not None:
            x1, y1, x2, y2 = self.roi_rect
            estimate_width = x2 - x1 + 1
            estimate_height = y2 - y1 + 1
            analysis_description = (
                f"ROI {estimate_width}x{estimate_height}"
            )
        else:
            estimate_width = width
            estimate_height = height
            analysis_description = (
                f"FULL IMAGE {width}x{height}"
            )

        extension = self.selected_image.suffix.lower()
        model_name = self.model_name()

        self.estimated_total_seconds = (
            self.performance_history.estimate_duration(
                model_name,
                extension,
                estimate_width,
                estimate_height
            )
        )

        self.run_start_time = datetime.now()

        estimated_end = (
            self.run_start_time
            + timedelta(
                seconds=self.estimated_total_seconds
            )
        )

        self.start_time_label.setText(
            "START TIME: "
            f"{self.run_start_time.strftime('%H:%M:%S')}"
        )

        self.elapsed_time_label.setText(
            "ELAPSED: 00:00"
        )

        self.remaining_time_label.setText(
            "ESTIMATED REMAINING: "
            f"{self.format_duration(self.estimated_total_seconds)}"
        )

        self.end_time_label.setText(
            "ESTIMATED END: "
            f"{estimated_end.strftime('%H:%M:%S')}"
        )

        self.estimate_basis_label.setText(
            "ESTIMATE BASIS: "
            f"{analysis_description} | "
            f"{estimate_width * estimate_height / 1_000_000:.2f} MP | "
            f"{model_name}"
        )

        self.timer.start()
        self.stage_log.clear()

        self.add_stage_message(
            "Starting leaf segmentation..."
        )

        if self.roi_rect is not None:
            self.add_stage_message(
                "Using selected ROI: "
                f"{self.roi_rect}"
            )
        else:
            self.add_stage_message(
                "Using full image."
            )

        self.status_label.setText("Starting...")
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        self.select_button.setEnabled(False)
        self.partial_region_button.setEnabled(False)
        self.clear_region_button.setEnabled(False)
        self.segment_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.log(
            "Starting segmentation: "
            f"{self.selected_image}"
        )

        self.thread = QThread()

        self.worker = SegmentationWorker(
            image_path=str(self.selected_image),
            model_path=str(MODEL_PATH),
            roi_rect=self.roi_rect
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.stage.connect(self.on_stage)
        self.worker.progress.connect(self.on_progress)
        self.worker.busy.connect(self.on_busy)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.cancelled.connect(self.on_cancelled)

        self.worker.finished.connect(
            self.thread.quit
        )
        self.worker.error.connect(
            self.thread.quit
        )
        self.worker.cancelled.connect(
            self.thread.quit
        )

        self.thread.finished.connect(
            self.worker.deleteLater
        )
        self.thread.finished.connect(
            self.thread_finished
        )

        self.thread.start()

    def model_name(self):
        return Path(MODEL_PATH).name

    def update_runtime_display(self):
        if self.run_start_time is None:
            return

        elapsed = (
            datetime.now()
            - self.run_start_time
        ).total_seconds()

        self.elapsed_time_label.setText(
            "ELAPSED: "
            f"{self.format_duration(elapsed)}"
        )

        if self.estimated_total_seconds is None:
            return

        remaining = (
            self.estimated_total_seconds
            - elapsed
        )

        remaining_display = (
            "00:00"
            if remaining < 0
            else self.format_duration(remaining)
        )

        estimated_end = (
            self.run_start_time
            + timedelta(
                seconds=self.estimated_total_seconds
            )
        )

        self.remaining_time_label.setText(
            "ESTIMATED REMAINING: "
            f"{remaining_display}"
        )

        self.end_time_label.setText(
            "ESTIMATED END: "
            f"{estimated_end.strftime('%H:%M:%S')}"
        )

    def stop_segmentation(self):
        if self.worker is None:
            return

        self.log(
            "User pressed STOP SEGMENTATION."
        )

        self.stop_button.setEnabled(False)
        self.status_label.setText(
            "STOP REQUESTED..."
        )

        self.add_stage_message(
            "STOP REQUESTED - "
            "waiting for current operation to finish..."
        )

        self.worker.request_stop()

    def on_stage(self, message):
        self.status_label.setText(message)
        self.add_stage_message(message)
        self.log(message)

    def on_progress(self, value):
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(value)

    def on_busy(self, is_busy):
        if is_busy:
            self.progress_bar.setMaximum(0)
            self.status_label.setText(
                "Running SAM 2 segmentation..."
            )
            self.add_stage_message(
                "Running SAM 2 segmentation..."
            )
            self.log(
                "SAM 2 inference is running..."
            )
        else:
            self.progress_bar.setMaximum(100)

    def on_finished(self, result):
        self.timer.stop()

        mask_count = result["mask_count"]
        width = result["width"]
        height = result["height"]

        if self.run_start_time is not None:
            duration = (
                datetime.now()
                - self.run_start_time
            ).total_seconds()
        else:
            duration = 0

        self.performance_history.save_run(
            model_name=self.model_name(),
            extension=self.selected_image.suffix.lower(),
            width=width,
            height=height,
            duration_seconds=duration
        )

        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)

        elapsed_display = self.format_duration(duration)

        message = (
            f"COMPLETED - {mask_count} masks"
        )

        self.status_label.setText(message)
        self.add_stage_message(message)

        self.add_stage_message(
            "TOTAL TIME: "
            f"{elapsed_display}"
        )

        if result.get("roi") is not None:
            self.add_stage_message(
                "ANALYZED ROI: "
                f"{result['roi']}"
            )

        self.add_stage_message(
            "DATABASE MEDIA ID: "
            f"{result.get('media_id')}"
        )

        self.add_stage_message(
            "DATABASE RUN ID: "
            f"{result.get('run_id')}"
        )

        self.add_stage_message(
            "RESULT DIRECTORY: "
            f"{result['run_directory']}"
        )

        self.elapsed_time_label.setText(
            "ELAPSED: "
            f"{elapsed_display}"
        )

        self.remaining_time_label.setText(
            "ESTIMATED REMAINING: 00:00"
        )

        self.end_time_label.setText(
            "FINISHED: "
            f"{datetime.now().strftime('%H:%M:%S')}"
        )

        self.log(message)

        QMessageBox.information(
            self,
            "LEAF SEGMENTATION",
            "Segmentation completed.\n\n"
            f"Masks: {mask_count}\n"
            f"Total time: {elapsed_display}\n"
            f"Database media ID: {result.get('media_id')}\n"
            f"Database run ID: {result.get('run_id')}\n\n"
            "Run directory:\n"
            f"{result['run_directory']}"
        )

    def on_cancelled(self):
        self.timer.stop()

        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        message = "SEGMENTATION STOPPED BY USER"

        self.status_label.setText(message)
        self.add_stage_message(message)

        if self.run_start_time is not None:
            elapsed = (
                datetime.now()
                - self.run_start_time
            ).total_seconds()

            self.elapsed_time_label.setText(
                "ELAPSED: "
                f"{self.format_duration(elapsed)}"
            )

        self.remaining_time_label.setText(
            "ESTIMATED REMAINING: STOPPED"
        )

        self.end_time_label.setText(
            "ESTIMATED END: STOPPED"
        )

        self.log(message)

    def on_error(self, error_type, error_details):
        self.timer.stop()

        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        self.status_label.setText(
            "SEGMENTATION FAILED"
        )

        self.add_stage_message(
            "SEGMENTATION FAILED"
        )
        self.add_stage_message(
            f"ERROR TYPE: {error_type}"
        )
        self.add_stage_message(error_details)

        QMessageBox.critical(
            self,
            "LEAF SEGMENTATION ERROR",
            f"ERROR TYPE:\n{error_type}\n\n"
            f"{error_details}"
        )

    def thread_finished(self):
        self.log(
            "Segmentation worker finished."
        )

        if self.thread is not None:
            self.thread.deleteLater()

        self.thread = None
        self.worker = None

        self.select_button.setEnabled(True)

        self.partial_region_button.setEnabled(
            self.selected_image is not None
        )

        self.segment_button.setEnabled(
            self.selected_image is not None
        )

        self.clear_region_button.setEnabled(
            self.roi_rect is not None
        )

        self.stop_button.setEnabled(False)

    def handle_error(self, title, exc):
        self.timer.stop()

        error_type = type(exc).__name__
        error_message = str(exc)
        full_traceback = traceback.format_exc()

        print("", flush=True)
        print("=" * 80, flush=True)
        print(title, flush=True)
        print(f"ERROR TYPE: {error_type}", flush=True)
        print(f"ERROR MESSAGE: {error_message}", flush=True)
        print("-" * 80, flush=True)
        print(full_traceback, flush=True)
        print("=" * 80, flush=True)

        self.status_label.setText("ERROR")

        self.add_stage_message(
            f"{title}: {error_message}"
        )

        QMessageBox.critical(
            self,
            title,
            f"ERROR TYPE:\n{error_type}\n\n"
            f"ERROR MESSAGE:\n{error_message}\n\n"
            f"FULL TRACEBACK:\n{full_traceback}"
        )

    def closeEvent(self, event):
        if self.thread is not None:
            reply = QMessageBox.question(
                self,
                "SEGMENTATION RUNNING",
                "Segmentation is currently running.\n\n"
                "Do you want to stop the operation?",
                (
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No
                )
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.stop_segmentation()

            event.ignore()
            return

        event.accept()
