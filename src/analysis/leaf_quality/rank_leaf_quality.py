from pathlib import Path
import cv2
import numpy as np


class LeafQualityRanker:
    """
    Rank segmented leaves by visual quality.

    The ranker does NOT remove leaves.
    It calculates quality measurements and returns the same leaf
    dictionaries enriched with quality information.

    Main quality factors:
        1. Sharpness
        2. Resolution / leaf pixel size
        3. Completeness
        4. Mask fill quality
        5. Edge truncation

    Expected leaf fields:
        mask, bbox, area, width, height

    Added fields:
        sharpness_score
        resolution_score
        completeness_score
        fill_score
        edge_score
        quality_score
        quality_rank
        quality_class
        is_reference
    """

    def __init__(
        self,
        reference_percent=30.0,
        min_reference_score=65.0,
        min_good_score=50.0,
    ):
        self.reference_percent = float(reference_percent)
        self.min_reference_score = float(min_reference_score)
        self.min_good_score = float(min_good_score)

    @staticmethod
    def _clip_score(value):
        return float(max(0.0, min(100.0, value)))

    @staticmethod
    def _percentile_score(value, values):
        """
        Convert a value to a relative 0-100 score using the distribution
        of the detected leaves. This makes the ranking work across images
        with different resolutions and numbers of leaves.
        """
        if not values:
            return 50.0

        values = np.asarray(values, dtype=np.float32)

        if len(values) == 1:
            return 100.0

        low = float(np.percentile(values, 10))
        high = float(np.percentile(values, 90))

        if high <= low:
            return 100.0

        score = (float(value) - low) / (high - low) * 100.0
        return LeafQualityRanker._clip_score(score)

    @staticmethod
    def _mask_crop(image, mask, bbox):
        x1, y1, x2, y2 = bbox

        h, w = image.shape[:2]

        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(x1, min(w - 1, int(x2)))
        y2 = max(y1, min(h - 1, int(y2)))

        crop = image[y1:y2 + 1, x1:x2 + 1]
        local_mask = mask[y1:y2 + 1, x1:x2 + 1]

        if crop.size == 0 or local_mask.size == 0:
            return None, None

        return crop, local_mask

    def _sharpness(self, image, mask, bbox):
        crop, local_mask = self._mask_crop(image, mask, bbox)

        if crop is None:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Use only pixels belonging to the leaf.
        leaf_pixels = gray[local_mask > 0]

        if leaf_pixels.size < 20:
            return 0.0

        # Laplacian variance is a useful relative sharpness measure.
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        values = laplacian[local_mask > 0]

        if values.size == 0:
            return 0.0

        return float(np.var(values))

    def _mask_fill_score(self, leaf):
        width = max(1, int(leaf.get("width", 1)))
        height = max(1, int(leaf.get("height", 1)))
        area = max(0, int(leaf.get("area", 0)))

        bbox_area = width * height

        if bbox_area <= 0:
            return 0.0

        fill_ratio = area / float(bbox_area)

        # A leaf normally occupies a meaningful part of its bounding box.
        # Very thin/fragmented masks are penalized.
        score = fill_ratio * 100.0

        return self._clip_score(score)

    def _edge_score(self, leaf, image_width, image_height):
        x1, y1, x2, y2 = leaf["bbox"]

        # Distance of the mask bbox from the image borders.
        distances = [
            max(0, int(x1)),
            max(0, int(y1)),
            max(0, int(image_width - 1 - x2)),
            max(0, int(image_height - 1 - y2)),
        ]

        min_distance = min(distances)

        # A leaf touching the image border is potentially truncated.
        if min_distance <= 0:
            return 20.0
        if min_distance <= 3:
            return 40.0
        if min_distance <= 8:
            return 60.0
        if min_distance <= 15:
            return 80.0

        return 100.0

    def _completeness_score(self, leaf, image_width, image_height):
        """
        Estimate completeness from geometry.

        This deliberately remains conservative: a leaf is not declared
        incomplete merely because it is small. Edge contact and unusual
        geometry contribute to the score.
        """
        edge_score = self._edge_score(
            leaf,
            image_width,
            image_height,
        )

        width = max(1, int(leaf.get("width", 1)))
        height = max(1, int(leaf.get("height", 1)))
        area = max(1, int(leaf.get("area", 1)))

        bbox_area = width * height
        fill_ratio = area / float(bbox_area)

        # Normal leaf masks tend to have a moderate/high bbox occupancy.
        if fill_ratio >= 0.55:
            shape_score = 100.0
        elif fill_ratio >= 0.40:
            shape_score = 85.0
        elif fill_ratio >= 0.28:
            shape_score = 65.0
        elif fill_ratio >= 0.18:
            shape_score = 45.0
        else:
            shape_score = 25.0

        return self._clip_score(
            edge_score * 0.55 + shape_score * 0.45
        )

    def rank(self, image, leaves):
        """
        Rank leaves without deleting any.

        Returns a new list ordered from highest quality to lowest quality.
        """
        if image is None:
            raise ValueError("image must not be None")

        if not leaves:
            return []

        image_height, image_width = image.shape[:2]

        sharpness_values = []

        for leaf in leaves:
            sharpness = self._sharpness(
                image,
                leaf["mask"],
                leaf["bbox"],
            )

            leaf["_raw_sharpness"] = sharpness
            sharpness_values.append(sharpness)

        area_values = [
            max(0, int(leaf.get("area", 0)))
            for leaf in leaves
        ]

        width_values = [
            max(0, int(leaf.get("width", 0)))
            for leaf in leaves
        ]

        height_values = [
            max(0, int(leaf.get("height", 0)))
            for leaf in leaves
        ]

        for leaf in leaves:
            sharpness_score = self._percentile_score(
                leaf["_raw_sharpness"],
                sharpness_values,
            )

            area_score = self._percentile_score(
                leaf.get("area", 0),
                area_values,
            )

            width_score = self._percentile_score(
                leaf.get("width", 0),
                width_values,
            )

            height_score = self._percentile_score(
                leaf.get("height", 0),
                height_values,
            )

            # Resolution is based mainly on actual leaf pixel area,
            # with width/height providing a secondary safeguard.
            resolution_score = (
                area_score * 0.70
                + width_score * 0.15
                + height_score * 0.15
            )

            fill_score = self._mask_fill_score(leaf)

            edge_score = self._edge_score(
                leaf,
                image_width,
                image_height,
            )

            completeness_score = self._completeness_score(
                leaf,
                image_width,
                image_height,
            )

            # Main ranking:
            # sharpness + resolution are intentionally dominant.
            quality_score = (
                sharpness_score * 0.40
                + resolution_score * 0.30
                + completeness_score * 0.20
                + fill_score * 0.07
                + edge_score * 0.03
            )

            leaf["sharpness_score"] = round(
                self._clip_score(sharpness_score), 2
            )
            leaf["resolution_score"] = round(
                self._clip_score(resolution_score), 2
            )
            leaf["completeness_score"] = round(
                self._clip_score(completeness_score), 2
            )
            leaf["fill_score"] = round(
                self._clip_score(fill_score), 2
            )
            leaf["edge_score"] = round(
                self._clip_score(edge_score), 2
            )
            leaf["quality_score"] = round(
                self._clip_score(quality_score), 2
            )

        # Highest quality first.
        ranked = sorted(
            leaves,
            key=lambda leaf: (
                leaf["quality_score"],
                leaf["sharpness_score"],
                leaf["resolution_score"],
                leaf["area"],
            ),
            reverse=True,
        )

        total = len(ranked)
        reference_count = max(
            1,
            int(np.ceil(total * self.reference_percent / 100.0)),
        )

        for rank_number, leaf in enumerate(
            ranked,
            start=1,
        ):
            leaf["quality_rank"] = rank_number

            is_reference = (
                rank_number <= reference_count
                and leaf["quality_score"]
                >= self.min_reference_score
            )

            leaf["is_reference"] = bool(is_reference)

            if is_reference:
                leaf["quality_class"] = "high_quality"
            elif leaf["quality_score"] >= self.min_good_score:
                leaf["quality_class"] = "partial_or_medium"
            else:
                leaf["quality_class"] = "low_quality"

            # The original segmentation ID is preserved.
            leaf["segmentation_id"] = leaf.get("id")

            # Remove internal calculation field.
            leaf.pop("_raw_sharpness", None)

        return ranked

    def rank_image_path(self, image_path, leaves):
        image_path = Path(image_path)

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Could not read image:\n{image_path}"
            )

        return self.rank(image, leaves)


def rank_leaf_quality(image, leaves):
    """
    Convenience function for the Paprika pipeline.
    """
    ranker = LeafQualityRanker()
    return ranker.rank(image, leaves)


if __name__ == "__main__":
    print(
        "LeafQualityRanker module loaded successfully."
    )
