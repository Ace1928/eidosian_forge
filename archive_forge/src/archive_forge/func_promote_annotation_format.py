import base64
import os
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from packaging import version
from .utils import (
from .utils.constants import (  # noqa: F401
def promote_annotation_format(annotation_format: Union[AnnotionFormat, AnnotationFormat]) -> AnnotationFormat:
    return AnnotationFormat(annotation_format.value)