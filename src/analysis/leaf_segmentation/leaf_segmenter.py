from pathlib import Path
import threading

import cv2
import numpy as np

from ultralytics import SAM


class SegmentationStopped(Exception):
    """Raised when the user requests a cooperative segmentation stop."""


class LeafSegmenter:
    """
    Leaf Segmenter V6

    Designed for dense paprika plants containing many overlapping
    and partially occluded leaves.

    Pipeline:
        1. Full-image SAM automatic segmentation
        2. Overlapping tile segmentation
        3. Dense interior-point discovery
        4. Independent SAM point prompts
        5. Candidate validation
        6. Clump detection
        7. Watershed splitting of dense clumps
        8. Small-leaf preservation
        9. Fast bbox pre-filter
        10. IoU + containment deduplication
        11. Spatial leaf numbering
        12. Original-image coordinates

    Public API remains compatible with the existing GUI:

        segmenter = LeafSegmenter(model_path)
        leaves = segmenter.segment(image_path)

    Every returned leaf contains a full-resolution mask in the
    ORIGINAL image coordinate system.
    """

    def __init__(
        self,
        model_path=None,
        tile_size=1024,
        tile_overlap=0.35,
        min_area_ratio=0.00002,
        max_area_ratio=0.30,
        duplicate_iou=0.82,
        containment=0.94,
        imgsz=1024,
        prompt_stride=72,
        max_prompt_points=220,
        split_min_area=700,
        split_min_seed_distance=24,
        sam_conf=0.10,
        max_tiles=16,
        max_total_prompt_points=240,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.model = None

        self._stop_event = threading.Event()

        self.tile_size = int(tile_size)
        self.tile_overlap = float(tile_overlap)
        self.max_tiles = int(max_tiles)
        self.max_total_prompt_points = int(max_total_prompt_points)

        self.last_tile_generation = {
            "generated": 0,
            "selected": 0,
            "skipped": 0,
        }

        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)

        self.duplicate_iou = float(duplicate_iou)
        self.containment = float(containment)

        self.imgsz = int(imgsz)
        self.sam_conf = float(sam_conf)

        self.prompt_stride = int(prompt_stride)
        self.max_prompt_points = int(max_prompt_points)

        self.split_min_area = int(split_min_area)
        self.split_min_seed_distance = int(split_min_seed_distance)

        self.clump_area_ratio = 0.008
        self.clump_min_area = max(
            1200,
            self.split_min_area,
        )

        self.parent_coverage_threshold = 0.42
        self.parent_min_children = 2

        self.diagnostics = {}

    def request_stop(self):
        self._stop_event.set()

        print(
            "[LEAF SEGMENTER V6] STOP requested.",
            flush=True,
        )

    def clear_stop(self):
        self._stop_event.clear()

    def is_stop_requested(self):
        return self._stop_event.is_set()

    def _check_stop(self):
        if self._stop_event.is_set():
            raise SegmentationStopped(
                "Segmentation stopped by user."
            )

    def load_model(self):
        if self.model_path is None:
            raise ValueError(
                "Leaf segmentation model path was not provided."
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Leaf segmentation model not found:\n"
                f"{self.model_path}"
            )

        print(
            "[LEAF SEGMENTER V6] "
            f"Loading SAM model: {self.model_path}",
            flush=True,
        )

        self.model = SAM(str(self.model_path))

        print(
            "[LEAF SEGMENTER V6] "
            "SAM model loaded successfully.",
            flush=True,
        )

    @staticmethod
    def clean_mask(mask_binary):
        mask_binary = (
            (mask_binary > 0).astype(np.uint8) * 255
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        mask_binary = cv2.morphologyEx(
            mask_binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        mask_binary = cv2.morphologyEx(
            mask_binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        return mask_binary

    @staticmethod
    def bbox_from_mask(mask):
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            return None

        return (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        )

    @staticmethod
    def bbox_iou(bbox_a, bbox_b):
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 < ix1 or iy2 < iy1:
            return 0.0

        intersection = (
            (ix2 - ix1 + 1)
            * (iy2 - iy1 + 1)
        )

        area_a = (
            (ax2 - ax1 + 1)
            * (ay2 - ay1 + 1)
        )

        area_b = (
            (bx2 - bx1 + 1)
            * (by2 - by1 + 1)
        )

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return float(intersection / union)

    @staticmethod
    def mask_iou(mask_a, mask_b):
        a = mask_a > 0
        b = mask_b > 0

        intersection = np.count_nonzero(a & b)
        union = np.count_nonzero(a | b)

        if union == 0:
            return 0.0

        return float(intersection / union)

    @staticmethod
    def mask_containment(mask_small, mask_large):
        a = mask_small > 0
        b = mask_large > 0

        small_area = np.count_nonzero(a)

        if small_area == 0:
            return 0.0

        intersection = np.count_nonzero(a & b)

        return float(intersection / small_area)

    def _tile_positions(self, width, height):
        tile_w = min(self.tile_size, width)
        tile_h = min(self.tile_size, height)

        step_x = max(
            1,
            int(tile_w * (1.0 - self.tile_overlap)),
        )

        step_y = max(
            1,
            int(tile_h * (1.0 - self.tile_overlap)),
        )

        xs = list(
            range(
                0,
                max(1, width - tile_w + 1),
                step_x,
            )
        )

        ys = list(
            range(
                0,
                max(1, height - tile_h + 1),
                step_y,
            )
        )

        last_x = max(0, width - tile_w)
        last_y = max(0, height - tile_h)

        if not xs or xs[-1] != last_x:
            xs.append(last_x)

        if not ys or ys[-1] != last_y:
            ys.append(last_y)

        tiles = []

        for y in ys:
            for x in xs:
                x2 = min(width, x + tile_w)
                y2 = min(height, y + tile_h)

                tiles.append(
                    (
                        x,
                        y,
                        x2,
                        y2,
                    )
                )

        generated_count = len(tiles)
        skipped_count = 0

        if len(tiles) > self.max_tiles:
            skipped_count = len(tiles) - self.max_tiles

            step = len(tiles) / float(self.max_tiles)

            selected = [
                tiles[
                    min(
                        int(i * step),
                        len(tiles) - 1,
                    )
                ]
                for i in range(self.max_tiles)
            ]

            unique = []
            seen = set()

            for tile in selected:
                if tile not in seen:
                    unique.append(tile)
                    seen.add(tile)

            tiles = unique

        self.last_tile_generation = {
            "generated": generated_count,
            "selected": len(tiles),
            "skipped": skipped_count,
        }

        return tiles

    def _run_sam(self, image):
        return self.model.predict(
            source=image,
            conf=self.sam_conf,
            verbose=False,
            imgsz=self.imgsz,
            retina_masks=True,
        )

    def _extract_candidates(
        self,
        results,
        full_width,
        full_height,
        offset_x=0,
        offset_y=0,
        source_pass="full",
        tile_bbox=None,
    ):
        candidates = []

        if not results:
            return candidates

        min_area = max(
            20,
            int(
                full_width
                * full_height
                * self.min_area_ratio
            ),
        )

        max_area = int(
            full_width
            * full_height
            * self.max_area_ratio
        )

        for result in results:
            if result.masks is None:
                continue

            masks = (
                result.masks.data
                .cpu()
                .numpy()
            )

            if tile_bbox is None:
                target_width = full_width
                target_height = full_height
            else:
                target_width = (
                    tile_bbox[2]
                    - tile_bbox[0]
                )

                target_height = (
                    tile_bbox[3]
                    - tile_bbox[1]
                )

            for source_index, mask in enumerate(masks):
                mask = cv2.resize(
                    mask,
                    (
                        target_width,
                        target_height,
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

                mask_binary = (
                    (mask > 0.5).astype(np.uint8)
                    * 255
                )

                mask_binary = self.clean_mask(
                    mask_binary
                )

                if offset_x != 0 or offset_y != 0:
                    full_mask = np.zeros(
                        (
                            full_height,
                            full_width,
                        ),
                        dtype=np.uint8,
                    )

                    mh, mw = mask_binary.shape

                    x2 = min(
                        full_width,
                        offset_x + mw,
                    )

                    y2 = min(
                        full_height,
                        offset_y + mh,
                    )

                    if (
                        x2 <= offset_x
                        or y2 <= offset_y
                    ):
                        continue

                    full_mask[
                        offset_y:y2,
                        offset_x:x2
                    ] = mask_binary[
                        :y2 - offset_y,
                        :x2 - offset_x
                    ]

                    mask_binary = full_mask

                area = int(
                    np.count_nonzero(mask_binary)
                )

                if area < min_area:
                    continue

                if area > max_area:
                    continue

                bbox = self.bbox_from_mask(
                    mask_binary
                )

                if bbox is None:
                    continue

                x1, y1, x2, y2 = bbox

                candidates.append(
                    {
                        "mask": mask_binary,
                        "bbox": bbox,
                        "area": area,
                        "width": x2 - x1 + 1,
                        "height": y2 - y1 + 1,
                        "source_index": source_index,
                        "source_pass": source_pass,
                        "tile_bbox": tile_bbox,
                        "prompt_score": 0.0,
                    }
                )

        return candidates

    @staticmethod
    def _green_region(image):
        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        lower = np.array(
            [20, 18, 12],
            dtype=np.uint8,
        )

        upper = np.array(
            [105, 255, 255],
            dtype=np.uint8,
        )

        green = cv2.inRange(
            hsv,
            lower,
            upper,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        green = cv2.morphologyEx(
            green,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        return green

    def _dense_leaf_points(
        self,
        image,
        existing_mask=None,
    ):
        green = self._green_region(image)

        if existing_mask is not None:
            existing_mask = (
                (existing_mask > 0).astype(np.uint8)
                * 255
            )

            green = cv2.bitwise_and(
                green,
                existing_mask,
            )

        binary = (green > 0).astype(np.uint8)

        if np.count_nonzero(binary) == 0:
            return []

        dist = cv2.distanceTransform(
            binary,
            cv2.DIST_L2,
            5,
        )

        maximum = float(dist.max())

        if maximum <= 0:
            return []

        peak_threshold = max(
            3.0,
            maximum * 0.18,
        )

        peaks = (
            (dist >= peak_threshold)
            .astype(np.uint8)
        )

        peaks = cv2.morphologyEx(
            peaks,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )

        count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                peaks,
                connectivity=8,
            )
        )

        points = []

        for index in range(1, count):
            area = stats[
                index,
                cv2.CC_STAT_AREA,
            ]

            if area < 3:
                continue

            cx, cy = centroids[index]

            ix = int(round(cx))
            iy = int(round(cy))

            if not (
                0 <= ix < dist.shape[1]
                and 0 <= iy < dist.shape[0]
            ):
                continue

            strength = float(dist[iy, ix])

            points.append(
                {
                    "x": ix,
                    "y": iy,
                    "strength": strength,
                }
            )

        points.sort(
            key=lambda p: p["strength"],
            reverse=True,
        )

        selected = []

        min_distance = max(
            12,
            int(self.prompt_stride * 0.25),
        )

        for point in points:
            too_close = False

            for existing in selected:
                dx = point["x"] - existing["x"]
                dy = point["y"] - existing["y"]

                distance_sq = (
                    dx * dx
                    + dy * dy
                )

                if distance_sq < min_distance * min_distance:
                    too_close = True
                    break

            if too_close:
                continue

            selected.append(point)

            if len(selected) >= self.max_prompt_points:
                break

        return selected

    def _run_single_point_prompt(
        self,
        image,
        point,
        source_pass="prompt",
        tile_bbox=None,
    ):
        height, width = image.shape[:2]

        x = float(point["x"])
        y = float(point["y"])

        try:
            results = self.model.predict(
                source=image,
                points=[x, y],
                labels=[1],
                verbose=False,
                imgsz=self.imgsz,
                retina_masks=True,
            )

        except Exception as exc:
            print(
                "[LEAF SEGMENTER V6] "
                f"Point prompt failed "
                f"({int(x)},{int(y)}): "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return []

        if not results:
            return []

        candidates = []

        for result in results:
            if result.masks is None:
                continue

            masks = (
                result.masks.data
                .cpu()
                .numpy()
            )

            scores = None

            if (
                hasattr(result, "boxes")
                and result.boxes is not None
                and hasattr(result.boxes, "conf")
            ):
                try:
                    scores = (
                        result.boxes.conf
                        .cpu()
                        .numpy()
                    )
                except Exception:
                    scores = None

            for mask_index, mask in enumerate(masks):
                mask = cv2.resize(
                    mask,
                    (
                        width,
                        height,
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

                mask_binary = (
                    (mask > 0.5).astype(np.uint8)
                    * 255
                )

                mask_binary = self.clean_mask(
                    mask_binary
                )

                point_x = int(round(x))
                point_y = int(round(y))

                if not (
                    0 <= point_x < width
                    and 0 <= point_y < height
                ):
                    continue

                if mask_binary[
                    point_y,
                    point_x
                ] == 0:
                    continue

                area = int(
                    np.count_nonzero(mask_binary)
                )

                min_area = max(
                    20,
                    int(
                        width
                        * height
                        * self.min_area_ratio
                    ),
                )

                max_area = int(
                    width
                    * height
                    * self.max_area_ratio
                )

                if area < min_area:
                    continue

                if area > max_area:
                    continue

                bbox = self.bbox_from_mask(
                    mask_binary
                )

                if bbox is None:
                    continue

                score = 0.0

                if (
                    scores is not None
                    and mask_index < len(scores)
                ):
                    score = float(
                        scores[mask_index]
                    )

                candidates.append(
                    {
                        "mask": mask_binary,
                        "bbox": bbox,
                        "area": area,
                        "width": bbox[2] - bbox[0] + 1,
                        "height": bbox[3] - bbox[1] + 1,
                        "source_index": mask_index,
                        "source_pass": source_pass,
                        "tile_bbox": tile_bbox,
                        "prompt_score": score,
                        "_prompt_x": point_x,
                        "_prompt_y": point_y,
                    }
                )

        candidates.sort(
            key=lambda c: (
                c["prompt_score"],
                -c["area"],
            ),
            reverse=True,
        )

        return candidates

    def _run_point_prompts(
        self,
        image,
        points,
        source_pass="prompt",
        tile_bbox=None,
    ):
        candidates = []

        for point_number, point in enumerate(
            points,
            start=1,
        ):
            self._check_stop()

            if point_number % 10 == 0:
                print(
                    "[LEAF SEGMENTER V6] "
                    f"Prompt {point_number}/"
                    f"{len(points)}",
                    flush=True,
                )

            point_candidates = (
                self._run_single_point_prompt(
                    image,
                    point,
                    source_pass=source_pass,
                    tile_bbox=tile_bbox,
                )
            )

            if not point_candidates:
                continue

            candidates.append(
                point_candidates[0]
            )

        return candidates

    def _is_suspicious_clump(
        self,
        candidate,
        image_width,
        image_height,
    ):
        image_area = image_width * image_height

        area_ratio = (
            candidate["area"]
            / float(image_area)
        )

        width = candidate["width"]
        height = candidate["height"]

        aspect = (
            max(width, height)
            / max(1, min(width, height))
        )

        if area_ratio > 0.020:
            return True

        if aspect > 4.5:
            return True

        if (
            width > image_width * 0.35
            and height > image_height * 0.18
        ):
            return True

        if (
            height > image_height * 0.35
            and width > image_width * 0.18
        ):
            return True

        return False

    def _clump_points(
        self,
        candidate,
        image,
    ):
        x1, y1, x2, y2 = candidate["bbox"]

        padding = 5

        rx1 = max(0, x1 - padding)
        ry1 = max(0, y1 - padding)

        rx2 = min(
            image.shape[1],
            x2 + padding + 1,
        )

        ry2 = min(
            image.shape[0],
            y2 + padding + 1,
        )

        region = image[
            ry1:ry2,
            rx1:rx2,
        ]

        local_mask = candidate["mask"][
            ry1:ry2,
            rx1:rx2,
        ]

        local_points = self._dense_leaf_points(
            region,
            local_mask,
        )

        points = []

        for point in local_points:
            points.append(
                {
                    "x": point["x"] + rx1,
                    "y": point["y"] + ry1,
                    "strength": point["strength"],
                }
            )

        return points

    def _watershed_split(
        self,
        mask,
        seed_points,
    ):
        if len(seed_points) < 2:
            return [mask]

        area = int(
            np.count_nonzero(mask > 0)
        )

        if area < self.split_min_area:
            return [mask]

        bbox = self.bbox_from_mask(mask)

        if bbox is None:
            return [mask]

        x1, y1, x2, y2 = bbox

        local_mask = mask[
            y1:y2 + 1,
            x1:x2 + 1,
        ]

        local_h, local_w = local_mask.shape

        markers = np.zeros(
            (
                local_h,
                local_w,
            ),
            dtype=np.int32,
        )

        marker_id = 1
        used = []

        for point in seed_points:
            px = int(point["x"]) - x1
            py = int(point["y"]) - y1

            if not (
                0 <= px < local_w
                and 0 <= py < local_h
            ):
                continue

            if local_mask[py, px] == 0:
                continue

            too_close = False

            for ux, uy in used:
                dx = px - ux
                dy = py - uy

                if (
                    dx * dx
                    + dy * dy
                    < self.split_min_seed_distance
                    * self.split_min_seed_distance
                ):
                    too_close = True
                    break

            if too_close:
                continue

            radius = max(
                2,
                min(
                    6,
                    int(point["strength"] * 0.18),
                ),
            )

            cv2.circle(
                markers,
                (px, py),
                radius,
                marker_id,
                -1,
            )

            used.append((px, py))
            marker_id += 1

        if marker_id <= 2:
            return [mask]

        dist = cv2.distanceTransform(
            (local_mask > 0).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )

        if dist.max() <= 0:
            return [mask]

        normalized = cv2.normalize(
            dist,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        watershed_gray = 255 - normalized

        pseudo = cv2.merge(
            [
                watershed_gray,
                watershed_gray,
                watershed_gray,
            ]
        )

        cv2.watershed(
            pseudo,
            markers,
        )

        pieces = []

        for label in range(1, marker_id):
            piece_local = np.zeros(
                (
                    local_h,
                    local_w,
                ),
                dtype=np.uint8,
            )

            piece_local[
                markers == label
            ] = 255

            piece_area = int(
                np.count_nonzero(piece_local)
            )

            if piece_area < max(
                80,
                int(self.split_min_area * 0.20),
            ):
                continue

            piece = np.zeros_like(
                mask,
                dtype=np.uint8,
            )

            piece[
                y1:y2 + 1,
                x1:x2 + 1
            ] = piece_local

            pieces.append(piece)

        if len(pieces) < 2:
            return [mask]

        return pieces

    def _is_reasonable_leaf(
        self,
        mask,
        image_shape,
        prompt_point=None,
    ):
        height, width = image_shape[:2]

        area = int(
            np.count_nonzero(mask > 0)
        )

        image_area = width * height

        min_area = max(
            80,
            int(
                image_area
                * self.min_area_ratio
            ),
        )

        max_area = int(
            image_area
            * self.max_area_ratio
        )

        if area < min_area:
            return False

        if area > max_area:
            return False

        bbox = self.bbox_from_mask(mask)

        if bbox is None:
            return False

        x1, y1, x2, y2 = bbox

        bw = x2 - x1 + 1
        bh = y2 - y1 + 1

        if bw < 8 or bh < 8:
            return False

        aspect = (
            max(bw, bh)
            / max(1, min(bw, bh))
        )

        if aspect > 8.0:
            return False

        if prompt_point is not None:
            px = int(prompt_point["x"])
            py = int(prompt_point["y"])

            if (
                0 <= px < width
                and 0 <= py < height
            ):
                if mask[py, px] == 0:
                    return False

        return True

    def _deduplicate(self, candidates):
        priority = {
            "prompt_clump": 5,
            "prompt": 4,
            "watershed": 4,
            "prompt_split": 4,
            "tile": 2,
            "full": 1,
        }

        candidates = sorted(
            candidates,
            key=lambda item: (
                priority.get(
                    item.get(
                        "source_pass",
                        "full",
                    ),
                    0,
                ),
                item.get(
                    "prompt_score",
                    0.0,
                ),
                item["area"],
            ),
            reverse=True,
        )

        accepted = []

        for candidate in candidates:
            duplicate = False

            for existing in accepted:
                bbox_overlap = self.bbox_iou(
                    candidate["bbox"],
                    existing["bbox"],
                )

                if bbox_overlap <= 0:
                    continue

                iou = self.mask_iou(
                    candidate["mask"],
                    existing["mask"],
                )

                if iou >= self.duplicate_iou:
                    duplicate = True
                    break

                area_small = min(
                    candidate["area"],
                    existing["area"],
                )

                area_large = max(
                    candidate["area"],
                    existing["area"],
                )

                area_ratio = (
                    area_small
                    / float(max(1, area_large))
                )

                if area_ratio >= 0.80:
                    containment_a = (
                        self.mask_containment(
                            candidate["mask"],
                            existing["mask"],
                        )
                    )

                    containment_b = (
                        self.mask_containment(
                            existing["mask"],
                            candidate["mask"],
                        )
                    )

                    if (
                        containment_a >= self.containment
                        or containment_b >= self.containment
                    ):
                        duplicate = True
                        break

            if not duplicate:
                accepted.append(candidate)

        return accepted

    def _build_leaves(self, candidates):
        candidates.sort(
            key=lambda item: (
                item["bbox"][1],
                item["bbox"][0],
            )
        )

        leaves = []

        for leaf_number, candidate in enumerate(
            candidates,
            start=1,
        ):
            mask = candidate["mask"]

            ys, xs = np.where(mask > 0)

            if len(xs) == 0:
                continue

            x1, y1, x2, y2 = candidate["bbox"]

            leaves.append(
                {
                    "id": leaf_number,
                    "source_index": (
                        candidate["source_index"] + 1
                    ),
                    "mask": mask,
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    "area": int(candidate["area"]),
                    "width": x2 - x1 + 1,
                    "height": y2 - y1 + 1,
                    "center_x": float(xs.mean()),
                    "center_y": float(ys.mean()),
                    "source_pass": candidate["source_pass"],
                    "tile_bbox": candidate.get("tile_bbox"),
                    "prompt_score": float(
                        candidate.get(
                            "prompt_score",
                            0.0,
                        )
                    ),
                }
            )

        for leaf_number, leaf in enumerate(
            leaves,
            start=1,
        ):
            leaf["id"] = leaf_number

        return leaves

    def _split_candidate_if_needed(
        self,
        candidate,
        image,
    ):
        self._check_stop()

        h, w = image.shape[:2]

        if not self._is_suspicious_clump(
            candidate,
            w,
            h,
        ):
            return [candidate]

        points = self._clump_points(
            candidate,
            image,
        )

        if len(points) < 2:
            return [candidate]

        pieces = self._watershed_split(
            candidate["mask"],
            points,
        )

        if len(pieces) < 2:
            return [candidate]

        result = []

        for index, piece in enumerate(pieces):
            self._check_stop()

            if not self._is_reasonable_leaf(
                piece,
                image.shape,
            ):
                continue

            bbox = self.bbox_from_mask(piece)

            if bbox is None:
                continue

            area = int(
                np.count_nonzero(piece)
            )

            child = dict(candidate)

            child.update(
                {
                    "mask": piece,
                    "bbox": bbox,
                    "area": area,
                    "width": bbox[2] - bbox[0] + 1,
                    "height": bbox[3] - bbox[1] + 1,
                    "source_pass": "watershed",
                    "source_index": index,
                }
            )

            result.append(child)

        if len(result) >= 2:
            self.diagnostics["clump_candidates"] += 1
            self.diagnostics["watershed_pieces"] += len(result)

            return result

        return [candidate]

    def _remove_explained_parent_clumps(
        self,
        candidates,
    ):
        if len(candidates) < 3:
            return candidates

        ordered = sorted(
            candidates,
            key=lambda c: c.get("area", 0),
            reverse=True,
        )

        removed = set()

        for i, parent in enumerate(ordered):
            self._check_stop()

            if (
                parent.get("area", 0)
                < self.clump_min_area
            ):
                continue

            parent_mask = parent["mask"] > 0

            parent_area = max(
                1,
                int(
                    np.count_nonzero(
                        parent_mask
                    )
                ),
            )

            coverage_mask = np.zeros_like(
                parent["mask"],
                dtype=np.uint8,
            )

            child_count = 0

            for j, child in enumerate(ordered):
                if i == j:
                    continue

                if (
                    child.get("area", 0)
                    >= parent.get("area", 0)
                ):
                    continue

                if (
                    self.bbox_iou(
                        parent["bbox"],
                        child["bbox"],
                    )
                    <= 0
                ):
                    continue

                if (
                    self.mask_containment(
                        child["mask"],
                        parent["mask"],
                    )
                    >= 0.70
                ):
                    coverage_mask[
                        child["mask"] > 0
                    ] = 255

                    child_count += 1

                if (
                    child_count
                    >= self.parent_min_children
                ):
                    coverage = (
                        np.count_nonzero(
                            (
                                coverage_mask > 0
                            )
                            & parent_mask
                        )
                        / float(parent_area)
                    )

                    if (
                        coverage
                        >= self.parent_coverage_threshold
                    ):
                        removed.add(id(parent))
                        break

        return [
            c
            for c in candidates
            if id(c) not in removed
        ]

    def segment(self, image_path):
        if self.model is None:
            self.load_model()

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                "Input image not found:\n"
                f"{image_path}"
            )

        print(
            "\n" + "=" * 72,
            flush=True,
        )

        print(
            "[LEAF SEGMENTER V6] START",
            flush=True,
        )

        print(
            "[LEAF SEGMENTER V6] "
            f"Image: {image_path}",
            flush=True,
        )

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                "Could not read input image:\n"
                f"{image_path}"
            )

        height, width = image.shape[:2]

        print(
            "[LEAF SEGMENTER V6] "
            f"Image size: {width} x {height}",
            flush=True,
        )

        self.diagnostics = {
            "full_candidates": 0,
            "tile_candidates": 0,
            "prompt_points": 0,
            "prompt_candidates": 0,
            "clump_candidates": 0,
            "watershed_pieces": 0,
            "tiles_generated": 0,
            "tiles_found": 0,
            "tiles_checked": 0,
            "tiles_remaining": 0,
            "tiles_skipped_by_limit": 0,
            "final_leaves": 0,
        }

        all_candidates = []
        total_prompt_points_used = 0

        self.clear_stop()

        print(
            "\n[LEAF SEGMENTER V6] "
            "PASS 1: FULL IMAGE",
            flush=True,
        )

        try:
            full_results = self._run_sam(image)

            full_candidates = self._extract_candidates(
                full_results,
                width,
                height,
                source_pass="full",
                tile_bbox=None,
            )

            all_candidates.extend(full_candidates)

            self.diagnostics["full_candidates"] = (
                len(full_candidates)
            )

            print(
                "[LEAF SEGMENTER V6] "
                f"Full candidates: {len(full_candidates)}",
                flush=True,
            )

        except Exception as exc:
            print(
                "[LEAF SEGMENTER V6] "
                f"Full pass failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

        tiles = self._tile_positions(
            width,
            height,
        )

        tile_info = self.last_tile_generation

        total_tiles_generated = tile_info["generated"]
        total_tiles_selected = tile_info["selected"]
        total_tiles_skipped = tile_info["skipped"]

        self.diagnostics["tiles_generated"] = (
            total_tiles_generated
        )

        self.diagnostics["tiles_found"] = (
            total_tiles_selected
        )

        self.diagnostics["tiles_checked"] = 0

        self.diagnostics["tiles_remaining"] = (
            total_tiles_selected
        )

        self.diagnostics["tiles_skipped_by_limit"] = (
            total_tiles_skipped
        )

        print(
            "\n[LEAF SEGMENTER V6] "
            "PASS 2: TILE PLAN",
            flush=True,
        )

        print(
            "[LEAF SEGMENTER V6] "
            f"TILE PLAN: "
            f"GENERATED={total_tiles_generated} | "
            f"FOUND={total_tiles_selected} | "
            f"SKIPPED_BY_LIMIT={total_tiles_skipped} | "
            f"CHECKED=0 | "
            f"REMAINING={total_tiles_selected}",
            flush=True,
        )

        for tile_number, (
            x1,
            y1,
            x2,
            y2,
        ) in enumerate(
            tiles,
            start=1,
        ):
            self._check_stop()

            print(
                "[LEAF SEGMENTER V6] "
                f"TILE {tile_number}/{total_tiles_selected} "
                f"START | "
                f"CHECKED={tile_number - 1} | "
                f"REMAINING="
                f"{total_tiles_selected - (tile_number - 1)} | "
                f"BOUNDS=({x1},{y1})-({x2},{y2})",
                flush=True,
            )

            tile = image[
                y1:y2,
                x1:x2,
            ]

            tile_candidates = []

            try:
                results = self._run_sam(tile)

                tile_candidates = self._extract_candidates(
                    results,
                    width,
                    height,
                    offset_x=x1,
                    offset_y=y1,
                    source_pass="tile",
                    tile_bbox=(
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                )

                all_candidates.extend(
                    tile_candidates
                )

                self.diagnostics["tile_candidates"] += (
                    len(tile_candidates)
                )

                print(
                    "[LEAF SEGMENTER V6] "
                    f"TILE {tile_number}/{total_tiles_selected} "
                    f"DONE | "
                    f"CHECKED={tile_number} | "
                    f"REMAINING="
                    f"{total_tiles_selected - tile_number} | "
                    f"CANDIDATES={len(tile_candidates)}",
                    flush=True,
                )

            except Exception as exc:
                print(
                    "[LEAF SEGMENTER V6] "
                    f"TILE {tile_number}/{total_tiles_selected} "
                    f"FAILED | "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

                print(
                    "[LEAF SEGMENTER V6] "
                    f"TILE {tile_number}/{total_tiles_selected} "
                    f"CHECKED={tile_number} | "
                    f"REMAINING="
                    f"{total_tiles_selected - tile_number}",
                    flush=True,
                )

            finally:
                self.diagnostics["tiles_checked"] = (
                    tile_number
                )

                self.diagnostics["tiles_remaining"] = (
                    total_tiles_selected - tile_number
                )

                print(
                    "[LEAF SEGMENTER V6] "
                    f"TILES STATUS: "
                    f"FOUND={total_tiles_selected} | "
                    f"CHECKED="
                    f"{self.diagnostics['tiles_checked']} | "
                    f"REMAINING="
                    f"{self.diagnostics['tiles_remaining']}",
                    flush=True,
                )

        print(
            "\n[LEAF SEGMENTER V6] "
            "TILES SUMMARY: "
            f"GENERATED={self.diagnostics['tiles_generated']} | "
            f"FOUND={self.diagnostics['tiles_found']} | "
            f"CHECKED={self.diagnostics['tiles_checked']} | "
            f"REMAINING={self.diagnostics['tiles_remaining']} | "
            f"SKIPPED_BY_LIMIT="
            f"{self.diagnostics['tiles_skipped_by_limit']}",
            flush=True,
        )

        print(
            "\n[LEAF SEGMENTER V6] "
            "PASS 3: DENSE POINT RECOVERY",
            flush=True,
        )

        for tile_number, (
            x1,
            y1,
            x2,
            y2,
        ) in enumerate(
            tiles,
            start=1,
        ):
            self._check_stop()

            tile = image[
                y1:y2,
                x1:x2,
            ]

            points = self._dense_leaf_points(tile)

            remaining_budget = (
                self.max_total_prompt_points
                - total_prompt_points_used
            )

            if remaining_budget <= 0:
                break

            points = points[
                :min(
                    len(points),
                    remaining_budget,
                )
            ]

            if not points:
                continue

            total_prompt_points_used += len(points)

            self.diagnostics["prompt_points"] += (
                len(points)
            )

            print(
                "[LEAF SEGMENTER V6] "
                f"Tile {tile_number}: "
                f"{len(points)} interior points",
                flush=True,
            )

            try:
                prompt_candidates = (
                    self._run_point_prompts(
                        tile,
                        points,
                        source_pass="prompt",
                        tile_bbox=(
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                    )
                )

                self.diagnostics["prompt_candidates"] += (
                    len(prompt_candidates)
                )

                for candidate in prompt_candidates:
                    local_mask = candidate["mask"]

                    full_mask = np.zeros(
                        (
                            height,
                            width,
                        ),
                        dtype=np.uint8,
                    )

                    mh, mw = local_mask.shape

                    target_y2 = min(
                        height,
                        y1 + mh,
                    )

                    target_x2 = min(
                        width,
                        x1 + mw,
                    )

                    if (
                        target_x2 <= x1
                        or target_y2 <= y1
                    ):
                        continue

                    full_mask[
                        y1:target_y2,
                        x1:target_x2
                    ] = local_mask[
                        :target_y2 - y1,
                        :target_x2 - x1
                    ]

                    candidate["mask"] = full_mask

                    bbox = self.bbox_from_mask(
                        full_mask
                    )

                    if bbox is None:
                        continue

                    candidate["bbox"] = bbox

                    candidate["area"] = int(
                        np.count_nonzero(full_mask)
                    )

                    candidate["width"] = (
                        bbox[2] - bbox[0] + 1
                    )

                    candidate["height"] = (
                        bbox[3] - bbox[1] + 1
                    )

                    all_candidates.append(candidate)

            except Exception as exc:
                print(
                    "[LEAF SEGMENTER V6] "
                    f"Prompt recovery failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        print(
            "\n[LEAF SEGMENTER V6] "
            "PASS 4: CLUMP SPLITTING",
            flush=True,
        )

        prefiltered = self._deduplicate(
            all_candidates
        )

        print(
            "[LEAF SEGMENTER V6] "
            f"Candidates before clump splitting: "
            f"{len(prefiltered)}",
            flush=True,
        )

        final_candidates = []

        for candidate_number, candidate in enumerate(
            prefiltered,
            start=1,
        ):
            self._check_stop()

            pieces = self._split_candidate_if_needed(
                candidate,
                image,
            )

            if len(pieces) > 1:
                print(
                    "[LEAF SEGMENTER V6] "
                    f"Candidate {candidate_number} "
                    f"split into {len(pieces)} leaves",
                    flush=True,
                )

            final_candidates.extend(pieces)

        final_candidates = (
            self._remove_explained_parent_clumps(
                final_candidates
            )
        )

        print(
            "\n[LEAF SEGMENTER V6] "
            "FINAL DEDUPLICATION",
            flush=True,
        )

        filtered = self._deduplicate(
            final_candidates
        )

        filtered = (
            self._remove_explained_parent_clumps(
                filtered
            )
        )

        leaves = self._build_leaves(
            filtered
        )

        self.diagnostics["final_leaves"] = (
            len(leaves)
        )

        print(
            "\n" + "=" * 72,
            flush=True,
        )

        print(
            "[LEAF SEGMENTER V6] FINAL RESULT",
            flush=True,
        )

        print(
            "[LEAF SEGMENTER V6] "
            f"LEAVES DETECTED: {len(leaves)}",
            flush=True,
        )

        print(
            "[LEAF SEGMENTER V6] "
            "Diagnostics:",
            flush=True,
        )

        for key, value in self.diagnostics.items():
            print(
                f"    {key}: {value}",
                flush=True,
            )

        print(
            "=" * 72,
            flush=True,
        )

        for leaf in leaves:
            print(
                "[LEAF SEGMENTER V6] "
                f"Leaf {leaf['id']:03d} "
                f"bbox={leaf['bbox']} "
                f"area={leaf['area']} "
                f"center=("
                f"{leaf['center_x']:.1f},"
                f"{leaf['center_y']:.1f}) "
                f"source={leaf['source_pass']}",
                flush=True,
            )

        return leaves


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    IMAGE_PATH = (
        PROJECT_ROOT
        / "images"
        / "test.jpg"
    )

    MODEL_CANDIDATES = [
        PROJECT_ROOT / "images" / "sam2.1_b.pt",
        PROJECT_ROOT / "models" / "sam2.1_b.pt",
        PROJECT_ROOT / "images" / "sam2_b.pt",
        PROJECT_ROOT / "models" / "sam2_b.pt",
    ]

    MODEL_PATH = next(
        (
            path
            for path in MODEL_CANDIDATES
            if path.exists()
        ),
        None,
    )

    if MODEL_PATH is None:
        raise FileNotFoundError(
            "Could not find SAM2 model.\n\n"
            "Checked:\n"
            + "\n".join(
                str(path)
                for path in MODEL_CANDIDATES
            )
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            "Test image not found:\n"
            f"{IMAGE_PATH}"
        )

    segmenter = LeafSegmenter(
        model_path=MODEL_PATH,
        tile_size=1024,
        tile_overlap=0.35,
        min_area_ratio=0.00002,
        max_area_ratio=0.30,
        duplicate_iou=0.82,
        containment=0.94,
        imgsz=1024,
        sam_conf=0.10,
        prompt_stride=72,
        max_prompt_points=220,
        split_min_area=700,
        split_min_seed_distance=24,
    )

    leaves = segmenter.segment(
        IMAGE_PATH
    )

    print(
        "\n" + "=" * 72,
        flush=True,
    )

    print(
        f"FINAL LEAF COUNT = {len(leaves)}",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )