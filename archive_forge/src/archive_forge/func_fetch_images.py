import copy
import json
import os
import warnings
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from .dynamic_module_utils import custom_object_save
from .feature_extraction_utils import BatchFeature as BaseBatchFeature
from .image_transforms import center_crop, normalize, rescale
from .image_utils import ChannelDimension
from .utils import (
def fetch_images(self, image_url_or_urls: Union[str, List[str]]):
    """
        Convert a single or a list of urls into the corresponding `PIL.Image` objects.

        If a single url is passed, the return value will be a single object. If a list is passed a list of objects is
        returned.
        """
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}
    if isinstance(image_url_or_urls, list):
        return [self.fetch_images(x) for x in image_url_or_urls]
    elif isinstance(image_url_or_urls, str):
        response = requests.get(image_url_or_urls, stream=True, headers=headers)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    else:
        raise ValueError(f'only a single or a list of entries is supported but got type={type(image_url_or_urls)}')