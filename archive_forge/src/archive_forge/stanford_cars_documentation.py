import pathlib
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from .utils import download_and_extract_archive, download_url, verify_str_arg
from .vision import VisionDataset
Returns pil_image and class_id for given index