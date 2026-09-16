from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QFrame,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
)


# ============================================================
# PAPRIKA PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_ROOT = PROJECT_ROOT / "results" / "segmentation"


# ============================================================
# HELPERS
# ============================================================

def get_creation_time(file_path):
    try:
        created_timestamp = file_path.stat().st_ctime

        return datetime.fromtimestamp(
            created_timestamp
        ).strftime(
            "%d/%m/%Y %H:%M:%S"
        )

    except Exception:
        return "UNKNOWN"


# ============================================================
# LEAF PREVIEW LABEL
# ============================================================

class LeafPreviewLabel(QLabel):

    double_clicked = Signal()

    def mouseDoubleClickEvent(self, event):

        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit()

        super().mouseDoubleClickEvent(event)


# ============================================================
# FULL IMAGE VIEWER
# ============================================================

class FullImageViewer(QWidget):

    def __init__(
        self,
        title_text,
        image_path,
        mask_path=None
    ):

        super().__init__()

        self.image_path = Path(image_path)

        self.mask_path = (
            Path(mask_path)
            if mask_path
            else None
        )

        self.setWindowTitle(
            title_text
        )

        self.resize(
            1200,
            850
        )

        self.create_ui()

    def create_ui(self):

        main_layout = QVBoxLayout(
            self
        )

        title = QLabel(
            self.image_path.name
        )

        title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        title.setStyleSheet(
            "font-size: 20px; font-weight: bold;"
        )

        main_layout.addWidget(
            title
        )

        info = QLabel(
            (
                f"CREATED: "
                f"{get_creation_time(self.image_path)}"
            )
        )

        info.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        main_layout.addWidget(
            info
        )

        image_scroll = QScrollArea()

        image_scroll.setWidgetResizable(
            False
        )

        image_label = QLabel()

        image_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        pixmap = QPixmap(
            str(self.image_path)
        )

        if pixmap.isNull():

            image_label.setText(
                (
                    "Could not load image:\n"
                    f"{self.image_path}"
                )
            )

        else:

            image_label.setPixmap(
                pixmap
            )

            image_label.adjustSize()

        image_scroll.setWidget(
            image_label
        )

        main_layout.addWidget(
            image_scroll,
            1
        )

        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        if (
            self.mask_path is not None
            and self.mask_path.exists()
        ):

            mask_title = QLabel(
                (
                    "MASK    |    CREATED: "
                    f"{get_creation_time(self.mask_path)}"
                )
            )

            mask_title.setAlignment(
                Qt.AlignmentFlag.AlignCenter
            )

            mask_title.setStyleSheet(
                "font-weight: bold;"
            )

            main_layout.addWidget(
                mask_title
            )

            mask_scroll = QScrollArea()

            mask_scroll.setWidgetResizable(
                False
            )

            mask_label = QLabel()

            mask_label.setAlignment(
                Qt.AlignmentFlag.AlignCenter
            )

            mask_pixmap = QPixmap(
                str(self.mask_path)
            )

            if mask_pixmap.isNull():

                mask_label.setText(
                    (
                        "Could not load mask:\n"
                        f"{self.mask_path}"
                    )
                )

            else:

                mask_label.setPixmap(
                    mask_pixmap
                )

                mask_label.adjustSize()

            mask_scroll.setWidget(
                mask_label
            )

            main_layout.addWidget(
                mask_scroll,
                1
            )


# ============================================================
# SEGMENTATION RESULTS - RUN LIST
# ============================================================

class SegmentationResults(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "PAPRIKA - SEGMENTATION RESULTS"
        )

        self.resize(
            1200,
            800
        )

        self.run_viewer = None

        self.create_ui()

        self.load_runs()

    def create_ui(self):

        main_layout = QVBoxLayout(
            self
        )

        title = QLabel(
            "SEGMENTATION RESULTS"
        )

        title.setStyleSheet(
            "font-size: 26px; font-weight: bold;"
        )

        main_layout.addWidget(
            title
        )

        runs_title = QLabel(
            "SEGMENTATION RUNS"
        )

        runs_title.setStyleSheet(
            "font-size: 18px; font-weight: bold;"
        )

        main_layout.addWidget(
            runs_title
        )

        self.run_list = QListWidget()

        self.run_list.setMinimumHeight(
            150
        )

        main_layout.addWidget(
            self.run_list
        )

        button_layout = QHBoxLayout()

        self.open_button = QPushButton(
            "OPEN SELECTED RUN"
        )

        self.open_button.setMinimumHeight(
            45
        )

        self.open_button.setEnabled(
            False
        )

        button_layout.addWidget(
            self.open_button
        )

        self.refresh_button = QPushButton(
            "REFRESH RUNS"
        )

        self.refresh_button.setMinimumHeight(
            45
        )

        button_layout.addWidget(
            self.refresh_button
        )

        main_layout.addLayout(
            button_layout
        )

        self.status_label = QLabel(
            "Ready"
        )

        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        main_layout.addWidget(
            self.status_label
        )

        self.run_list.itemSelectionChanged.connect(
            self.on_run_selected
        )

        self.run_list.itemDoubleClicked.connect(
            self.open_selected_run
        )

        self.open_button.clicked.connect(
            self.open_selected_run
        )

        self.refresh_button.clicked.connect(
            self.load_runs
        )

    # --------------------------------------------------------
    # LOAD RUNS
    # --------------------------------------------------------

    def load_runs(self):

        self.run_list.clear()

        self.open_button.setEnabled(
            False
        )

        if not RESULTS_ROOT.exists():

            self.status_label.setText(
                (
                    "No segmentation runs found.\n"
                    f"Results path: {RESULTS_ROOT}"
                )
            )

            return

        run_directories = sorted(
            [
                path
                for path in RESULTS_ROOT.iterdir()
                if path.is_dir()
            ],
            key=lambda path: path.stat().st_ctime,
            reverse=True
        )

        if not run_directories:

            self.status_label.setText(
                "No segmentation runs found."
            )

            return

        for run_directory in run_directories:

            item_text = (
                f"{run_directory.name}"
                "    |    CREATED: "
                f"{get_creation_time(run_directory)}"
            )

            item = QListWidgetItem(
                item_text
            )

            item.setData(
                Qt.ItemDataRole.UserRole,
                str(run_directory)
            )

            self.run_list.addItem(
                item
            )

        self.status_label.setText(
            (
                f"Found "
                f"{len(run_directories)} "
                f"segmentation runs."
            )
        )

    # --------------------------------------------------------
    # RUN SELECTION
    # --------------------------------------------------------

    def on_run_selected(self):

        self.open_button.setEnabled(
            self.run_list.currentItem() is not None
        )

    # --------------------------------------------------------
    # OPEN RUN
    # --------------------------------------------------------

    def open_selected_run(self):

        item = self.run_list.currentItem()

        if item is None:
            return

        run_directory = item.data(
            Qt.ItemDataRole.UserRole
        )

        self.run_viewer = SegmentationRunViewer(
            Path(run_directory)
        )

        self.run_viewer.show()

        self.run_viewer.raise_()

        self.run_viewer.activateWindow()


# ============================================================
# SEGMENTATION RUN VIEWER
# ============================================================

class SegmentationRunViewer(QWidget):

    def __init__(
        self,
        run_directory
    ):

        super().__init__()

        self.run_directory = Path(
            run_directory
        )

        self.full_viewer = None

        self.setWindowTitle(
            (
                "PAPRIKA - SEGMENTATION RUN - "
                f"{self.run_directory.name}"
            )
        )

        self.resize(
            1400,
            900
        )

        self.create_ui()

        self.load_run()

    # --------------------------------------------------------
    # CREATE UI
    # --------------------------------------------------------

    def create_ui(self):

        main_layout = QVBoxLayout(
            self
        )

        title = QLabel(
            (
                "SEGMENTATION RUN\n"
                f"{self.run_directory.name}"
            )
        )

        title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        title.setStyleSheet(
            "font-size: 22px; font-weight: bold;"
        )

        main_layout.addWidget(
            title
        )

        run_info = QLabel(
            (
                "RUN CREATED: "
                f"{get_creation_time(self.run_directory)}"
            )
        )

        run_info.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        main_layout.addWidget(
            run_info
        )

        main_layout.addWidget(
            QLabel(
                "SOURCE / SELECTED REGION"
            )
        )

        top_row = QHBoxLayout()

        original_panel = self.create_image_panel(
            "ORIGINAL IMAGE",
            500
        )

        self.original_image_label = (
            original_panel[0]
        )

        self.original_info_label = (
            original_panel[1]
        )

        top_row.addWidget(
            original_panel[2],
            1
        )

        region_panel = self.create_image_panel(
            "SELECTED REGION / ROI",
            500
        )

        self.region_image_label = (
            region_panel[0]
        )

        self.region_info_label = (
            region_panel[1]
        )

        top_row.addWidget(
            region_panel[2],
            1
        )

        main_layout.addLayout(
            top_row,
            3
        )

        # ----------------------------------------------------
        # LEAVES
        # ----------------------------------------------------

        leaves_title = QLabel(
            "LEAVES"
        )

        leaves_title.setStyleSheet(
            "font-size: 18px; font-weight: bold;"
        )

        main_layout.addWidget(
            leaves_title
        )

        self.leaves_scroll = QScrollArea()

        self.leaves_scroll.setWidgetResizable(
            True
        )

        self.leaves_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        self.leaves_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.leaves_container = QWidget()

        self.leaves_layout = QHBoxLayout(
            self.leaves_container
        )

        self.leaves_layout.setAlignment(
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignTop
        )

        self.leaves_layout.setSpacing(
            18
        )

        self.leaves_scroll.setWidget(
            self.leaves_container
        )

        main_layout.addWidget(
            self.leaves_scroll,
            2
        )

        hint = QLabel(
            "Double-click any leaf to open it at full size."
        )

        hint.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        main_layout.addWidget(
            hint
        )

    # --------------------------------------------------------
    # IMAGE PANEL
    # --------------------------------------------------------

    def create_image_panel(
        self,
        title_text,
        max_width
    ):

        panel = QFrame()

        panel.setFrameShape(
            QFrame.Shape.Box
        )

        panel.setMinimumWidth(
            520
        )

        layout = QVBoxLayout(
            panel
        )

        title = QLabel(
            title_text
        )

        title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        title.setStyleSheet(
            "font-size: 17px; font-weight: bold;"
        )

        layout.addWidget(
            title
        )

        image_label = QLabel()

        image_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        image_label.setMinimumHeight(
            280
        )

        image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        image_label.setStyleSheet(
            "border: 1px solid gray;"
        )

        layout.addWidget(
            image_label,
            1
        )

        info_label = QLabel()

        info_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        layout.addWidget(
            info_label
        )

        return (
            image_label,
            info_label,
            panel
        )

    # --------------------------------------------------------
    # LOAD RUN
    # --------------------------------------------------------

    def load_run(self):

        original_directory = (
            self.run_directory
            / "original"
        )

        roi_directory = (
            self.run_directory
            / "roi"
        )

        leaves_directory = (
            self.run_directory
            / "leaves"
        )

        masks_directory = (
            self.run_directory
            / "masks"
        )

        # ----------------------------------------------------
        # ORIGINAL
        # ----------------------------------------------------

        original_files = (
            sorted(
                [
                    path
                    for path in original_directory.iterdir()
                    if path.is_file()
                ]
            )
            if original_directory.exists()
            else []
        )

        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        roi_files = (
            sorted(
                [
                    path
                    for path in roi_directory.iterdir()
                    if path.is_file()
                ]
            )
            if roi_directory.exists()
            else []
        )

        if original_files:

            original_path = original_files[0]

            self.show_image(
                self.original_image_label,
                original_path,
                700,
                360
            )

            self.original_info_label.setText(
                (
                    f"FILE: {original_path.name}"
                    "    |    CREATED: "
                    f"{get_creation_time(original_path)}"
                )
            )

        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        if roi_files:

            roi_path = roi_files[0]

            self.show_image(
                self.region_image_label,
                roi_path,
                700,
                360
            )

            self.region_info_label.setText(
                (
                    f"FILE: {roi_path.name}"
                    "    |    CREATED: "
                    f"{get_creation_time(roi_path)}"
                )
            )

        else:

            self.region_image_label.setText(
                "FULL IMAGE - NO ROI SELECTED"
            )

            if original_files:

                self.region_info_label.setText(
                    (
                        "Analysis region: FULL IMAGE"
                        "    |    CREATED: "
                        f"{get_creation_time(original_files[0])}"
                    )
                )

        # ----------------------------------------------------
        # LEAVES
        # ----------------------------------------------------

        leaf_files = (
            sorted(
                leaves_directory.glob(
                    "leaf_*.png"
                ),
                key=lambda path: path.name
            )
            if leaves_directory.exists()
            else []
        )

        # ----------------------------------------------------
        # MASKS
        # ----------------------------------------------------

        mask_files = (
            sorted(
                masks_directory.glob(
                    "mask_*.png"
                ),
                key=lambda path: path.name
            )
            if masks_directory.exists()
            else []
        )

        mask_map = {
            path.stem.replace(
                "mask_",
                ""
            ): path

            for path in mask_files
        }

        # ----------------------------------------------------
        # NO LEAVES
        # ----------------------------------------------------

        if not leaf_files:

            message = QLabel(
                (
                    "No individual leaf results found.\n\n"
                    f"Expected folder:\n"
                    f"{leaves_directory}"
                )
            )

            message.setAlignment(
                Qt.AlignmentFlag.AlignCenter
            )

            self.leaves_layout.addWidget(
                message
            )

            return

        # ----------------------------------------------------
        # ADD LEAF CARDS
        # ----------------------------------------------------

        for leaf_path in leaf_files:

            leaf_id = leaf_path.stem.replace(
                "leaf_",
                ""
            )

            mask_path = mask_map.get(
                leaf_id
            )

            self.add_leaf_card(
                leaf_id,
                leaf_path,
                mask_path
            )

        self.leaves_layout.addStretch(
            1
        )

    # --------------------------------------------------------
    # SHOW IMAGE
    # --------------------------------------------------------

    def show_image(
        self,
        label,
        image_path,
        width,
        height
    ):

        pixmap = QPixmap(
            str(image_path)
        )

        if pixmap.isNull():

            label.setText(
                (
                    "Could not load image:\n"
                    f"{image_path}"
                )
            )

            return

        scaled = pixmap.scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        label.setPixmap(
            scaled
        )

    # --------------------------------------------------------
    # LEAF CARD
    # --------------------------------------------------------

    def add_leaf_card(
        self,
        leaf_id,
        leaf_path,
        mask_path
    ):

        frame = QFrame()

        frame.setFrameShape(
            QFrame.Shape.Box
        )

        frame.setMinimumWidth(
            230
        )

        frame.setMaximumWidth(
            260
        )

        layout = QVBoxLayout(
            frame
        )

        title = QLabel(
            f"LEAF {leaf_id}"
        )

        title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        title.setStyleSheet(
            "font-size: 16px; font-weight: bold;"
        )

        layout.addWidget(
            title
        )

        image_label = LeafPreviewLabel()

        image_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        image_label.setFixedSize(
            220,
            190
        )

        image_label.setStyleSheet(
            "border: 1px solid gray;"
        )

        self.show_image(
            image_label,
            leaf_path,
            210,
            180
        )

        image_label.double_clicked.connect(
            lambda: self.open_leaf_full_size(
                leaf_path,
                mask_path,
                leaf_id
            )
        )

        layout.addWidget(
            image_label
        )

        created_label = QLabel(
            (
                "CREATED: "
                f"{get_creation_time(leaf_path)}"
            )
        )

        created_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        layout.addWidget(
            created_label
        )

        double_click_label = QLabel(
            "DOUBLE-CLICK TO OPEN"
        )

        double_click_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        layout.addWidget(
            double_click_label
        )

        self.leaves_layout.addWidget(
            frame
        )

    # --------------------------------------------------------
    # FULL SIZE LEAF
    # --------------------------------------------------------

    def open_leaf_full_size(
        self,
        leaf_path,
        mask_path,
        leaf_id
    ):

        self.full_viewer = FullImageViewer(
            (
                "PAPRIKA - LEAF "
                f"{leaf_id} - FULL SIZE"
            ),
            leaf_path,
            mask_path
        )

        self.full_viewer.show()

        self.full_viewer.raise_()

        self.full_viewer.activateWindow()

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def closeEvent(
        self,
        event
    ):

        if self.full_viewer is not None:

            self.full_viewer.close()

            self.full_viewer = None

        event.accept()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    app = QApplication([])

    window = SegmentationResults()

    window.show()

    app.exec()