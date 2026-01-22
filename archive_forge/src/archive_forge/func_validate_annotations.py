import base64
import os
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from packaging import version
from .utils import (
from .utils.constants import (  # noqa: F401
def validate_annotations(annotation_format: AnnotationFormat, supported_annotation_formats: Tuple[AnnotationFormat, ...], annotations: List[Dict]) -> None:
    if isinstance(annotation_format, AnnotionFormat):
        logger.warning_once(f'`{annotation_format.__class__.__name__}` is deprecated and will be removed in v4.38. Please use `{AnnotationFormat.__name__}` instead.')
        annotation_format = promote_annotation_format(annotation_format)
    if annotation_format not in supported_annotation_formats:
        raise ValueError(f'Unsupported annotation format: {format} must be one of {supported_annotation_formats}')
    if annotation_format is AnnotationFormat.COCO_DETECTION:
        if not valid_coco_detection_annotations(annotations):
            raise ValueError('Invalid COCO detection annotations. Annotations must a dict (single image) or list of dicts (batch of images) with the following keys: `image_id` and `annotations`, with the latter being a list of annotations in the COCO format.')
    if annotation_format is AnnotationFormat.COCO_PANOPTIC:
        if not valid_coco_panoptic_annotations(annotations):
            raise ValueError('Invalid COCO panoptic annotations. Annotations must a dict (single image) or list of dicts (batch of images) with the following keys: `image_id`, `file_name` and `segments_info`, with the latter being a list of annotations in the COCO format.')