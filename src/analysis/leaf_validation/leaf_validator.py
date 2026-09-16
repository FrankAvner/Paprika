from pathlib import Path

import cv2
import numpy as np


class LeafValidator:
    """
    External validation layer for paprika leaf candidates.

    Purpose:
        Reject obvious non-leaf regions such as:
            - stems / stalks
            - very thin elongated structures
            - large unstructured green areas
            - uniform green background patches

    Important:
        Green color alone is NOT a reason to reject a candidate.
        Real paprika leaves are naturally green.

    The validator uses several independent signals:
        1. Geometry
        2. Solidity
        3. Extent
        4. Thickness
        5. Edge density
        6. Color variation
        7. Green coverage
        8. Connected-component structure

    The validator is intentionally conservative.
    """

    def __init__(
        self,
        enabled=True,

        # --------------------------------------------------------------
        # STEM DETECTION
        # --------------------------------------------------------------
        min_width=8,
        min_thickness=3.0,
        stem_aspect_ratio=7.5,
        stem_solidity=0.72,

        # --------------------------------------------------------------
        # GREEN REGION DETECTION
        # --------------------------------------------------------------
        green_ratio_threshold=0.985,
        green_uniformity_threshold=0.035,
        green_edge_density_threshold=0.012,

        # --------------------------------------------------------------
        # LEAF SHAPE
        # --------------------------------------------------------------
        min_solidity=0.18,
        min_extent=0.18,

        # --------------------------------------------------------------
        # TEXTURE / EDGE
        # --------------------------------------------------------------
        min_edge_density=0.006,
        min_color_std=4.0,

        # --------------------------------------------------------------
        # VERY LARGE UNSTRUCTURED REGIONS
        # --------------------------------------------------------------
        large_region_ratio=0.035,
        large_region_edge_density=0.018,
        large_region_color_std=6.0,
    ):
        self.enabled = bool(enabled)

        self.min_width = float(min_width)
        self.min_thickness = float(
            min_thickness
        )

        self.stem_aspect_ratio = float(
            stem_aspect_ratio
        )

        self.stem_solidity = float(
            stem_solidity
        )

        self.green_ratio_threshold = float(
            green_ratio_threshold
        )

        self.green_uniformity_threshold = float(
            green_uniformity_threshold
        )

        self.green_edge_density_threshold = float(
            green_edge_density_threshold
        )

        self.min_solidity = float(
            min_solidity
        )

        self.min_extent = float(
            min_extent
        )

        self.min_edge_density = float(
            min_edge_density
        )

        self.min_color_std = float(
            min_color_std
        )

        self.large_region_ratio = float(
            large_region_ratio
        )

        self.large_region_edge_density = float(
            large_region_edge_density
        )

        self.large_region_color_std = float(
            large_region_color_std
        )

        self.last_reason = None
        self.last_metrics = {}

    # ------------------------------------------------------------------
    # BASIC HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_area(mask):
        return int(
            np.count_nonzero(mask > 0)
        )

    @staticmethod
    def _bbox(mask):
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
    def _contour(mask):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        return max(
            contours,
            key=cv2.contourArea,
        )

    # ------------------------------------------------------------------
    # GEOMETRY
    # ------------------------------------------------------------------

    def _geometry_metrics(
        self,
        mask,
        image_shape,
    ):
        height, width = image_shape[:2]

        area = self._mask_area(mask)

        if area <= 0:
            return None

        bbox = self._bbox(mask)

        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox

        bw = x2 - x1 + 1
        bh = y2 - y1 + 1

        bbox_area = max(
            1,
            bw * bh,
        )

        aspect = (
            max(bw, bh)
            / max(
                1,
                min(bw, bh),
            )
        )

        contour = self._contour(mask)

        solidity = 0.0
        contour_area = 0.0

        if contour is not None:
            contour_area = float(
                cv2.contourArea(contour)
            )

            hull = cv2.convexHull(
                contour
            )

            hull_area = float(
                cv2.contourArea(hull)
            )

            if hull_area > 0:
                solidity = (
                    contour_area
                    / hull_area
                )

        extent = (
            area
            / float(bbox_area)
        )

        distance = cv2.distanceTransform(
            (mask > 0).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )

        max_thickness = float(
            distance.max() * 2.0
        )

        area_ratio = (
            area
            / float(
                max(
                    1,
                    width * height,
                )
            )
        )

        return {
            "area": area,
            "area_ratio": area_ratio,
            "bbox_width": bw,
            "bbox_height": bh,
            "aspect_ratio": aspect,
            "solidity": solidity,
            "extent": extent,
            "max_thickness": max_thickness,
            "contour_area": contour_area,
        }

    # ------------------------------------------------------------------
    # COLOR / TEXTURE
    # ------------------------------------------------------------------

    def _appearance_metrics(
        self,
        image,
        mask,
    ):
        binary = (
            mask > 0
        ).astype(np.uint8)

        area = int(
            np.count_nonzero(binary)
        )

        if area <= 0:
            return None

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        )

        green_mask = cv2.inRange(
            hsv,
            np.array(
                [20, 25, 15],
                dtype=np.uint8,
            ),
            np.array(
                [105, 255, 255],
                dtype=np.uint8,
            ),
        )

        green_inside = cv2.bitwise_and(
            green_mask,
            green_mask,
            mask=binary,
        )

        green_ratio = (
            np.count_nonzero(
                green_inside
            )
            / float(area)
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        edges = cv2.Canny(
            gray,
            50,
            150,
        )

        edges_inside = cv2.bitwise_and(
            edges,
            edges,
            mask=binary,
        )

        edge_density = (
            np.count_nonzero(
                edges_inside
            )
            / float(area)
        )

        hsv_pixels = hsv[
            binary > 0
        ]

        lab_pixels = lab[
            binary > 0
        ]

        if len(hsv_pixels) > 0:
            hue_std = float(
                np.std(
                    hsv_pixels[:, 0]
                )
            )

            saturation_std = float(
                np.std(
                    hsv_pixels[:, 1]
                )
            )

            value_std = float(
                np.std(
                    hsv_pixels[:, 2]
                )
            )
        else:
            hue_std = 0.0
            saturation_std = 0.0
            value_std = 0.0

        if len(lab_pixels) > 0:
            color_std = float(
                np.mean(
                    [
                        np.std(
                            lab_pixels[:, 0]
                        ),
                        np.std(
                            lab_pixels[:, 1]
                        ),
                        np.std(
                            lab_pixels[:, 2]
                        ),
                    ]
                )
            )
        else:
            color_std = 0.0

        return {
            "green_ratio": green_ratio,
            "edge_density": edge_density,
            "hue_std": hue_std,
            "saturation_std": saturation_std,
            "value_std": value_std,
            "color_std": color_std,
        }

    # ------------------------------------------------------------------
    # STEM DETECTION
    # ------------------------------------------------------------------

    def _is_stem(
        self,
        geometry,
    ):
        if geometry is None:
            return False

        aspect = geometry[
            "aspect_ratio"
        ]

        thickness = geometry[
            "max_thickness"
        ]

        width = geometry[
            "bbox_width"
        ]

        height = geometry[
            "bbox_height"
        ]

        solidity = geometry[
            "solidity"
        ]

        if (
            width < self.min_width
            and height < self.min_width
        ):
            return True

        # Extremely elongated and thin object.
        if (
            aspect >= self.stem_aspect_ratio
            and thickness <= self.min_thickness
        ):
            return True

        # Long narrow object with weak leaf-like solidity.
        if (
            aspect >= self.stem_aspect_ratio
            and solidity < self.stem_solidity
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # UNSTRUCTURED GREEN REGION
    # ------------------------------------------------------------------

    def _is_unstructured_green(
        self,
        geometry,
        appearance,
    ):
        if (
            geometry is None
            or appearance is None
        ):
            return False

        green_ratio = appearance[
            "green_ratio"
        ]

        edge_density = appearance[
            "edge_density"
        ]

        color_std = appearance[
            "color_std"
        ]

        area_ratio = geometry[
            "area_ratio"
        ]

        # Do not reject merely because the candidate is green.
        #
        # Reject only when the candidate is:
        #   almost completely green
        #   AND visually uniform
        #   AND has very few internal edges.
        #
        if (
            green_ratio
            >= self.green_ratio_threshold
            and color_std
            <= self.green_uniformity_threshold
            and edge_density
            <= self.green_edge_density_threshold
        ):
            return True

        # Large green areas receive an additional conservative test.
        if (
            area_ratio
            >= self.large_region_ratio
            and green_ratio
            >= 0.95
            and edge_density
            <= self.large_region_edge_density
            and color_std
            <= self.large_region_color_std
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # PUBLIC VALIDATION
    # ------------------------------------------------------------------

    def validate(
        self,
        image,
        mask,
        prompt_point=None,
    ):
        """
        Return True when the candidate is sufficiently leaf-like.

        This method intentionally errs on the side of keeping
        uncertain candidates.
        """

        self.last_reason = None
        self.last_metrics = {}

        if not self.enabled:
            return True

        if image is None:
            return True

        if mask is None:
            self.last_reason = "empty_mask"
            return False

        binary = (
            (mask > 0)
            .astype(np.uint8)
            * 255
        )

        area = self._mask_area(
            binary
        )

        if area <= 0:
            self.last_reason = "empty_mask"
            return False

        geometry = self._geometry_metrics(
            binary,
            image.shape,
        )

        appearance = self._appearance_metrics(
            image,
            binary,
        )

        if (
            geometry is None
            or appearance is None
        ):
            self.last_reason = (
                "metrics_failed"
            )
            return False

        self.last_metrics = {
            **geometry,
            **appearance,
        }

        # --------------------------------------------------------------
        # Prompt must remain inside mask.
        # --------------------------------------------------------------

        if prompt_point is not None:

            px = int(
                prompt_point.get(
                    "x",
                    -1,
                )
            )

            py = int(
                prompt_point.get(
                    "y",
                    -1,
                )
            )

            h, w = image.shape[:2]

            if (
                0 <= px < w
                and 0 <= py < h
            ):
                if (
                    binary[py, px]
                    == 0
                ):
                    self.last_reason = (
                        "prompt_outside_mask"
                    )
                    return False

        # --------------------------------------------------------------
        # Obvious stem.
        # --------------------------------------------------------------

        if self._is_stem(
            geometry
        ):
            self.last_reason = "stem"
            return False

        # --------------------------------------------------------------
        # Very poor geometric structure.
        #
        # We deliberately do NOT reject based only on aspect ratio.
        # Paprika leaves can naturally be elongated.
        # --------------------------------------------------------------

        if (
            geometry["solidity"]
            < self.min_solidity
            and geometry["extent"]
            < self.min_extent
        ):
            self.last_reason = (
                "poor_leaf_geometry"
            )
            return False

        # --------------------------------------------------------------
        # Unstructured green region.
        # --------------------------------------------------------------

        if self._is_unstructured_green(
            geometry,
            appearance,
        ):
            self.last_reason = (
                "unstructured_green_region"
            )
            return False

        # --------------------------------------------------------------
        # Very uniform candidate.
        #
        # Only reject if it is also geometrically suspicious.
        # This protects real smooth leaves.
        # --------------------------------------------------------------

        if (
            appearance["edge_density"]
            < self.min_edge_density
            and appearance["color_std"]
            < self.min_color_std
            and geometry["solidity"]
            < 0.55
            and geometry["extent"]
            < 0.45
        ):
            self.last_reason = (
                "uniform_non_leaf_region"
            )
            return False

        self.last_reason = "accepted"

        return True