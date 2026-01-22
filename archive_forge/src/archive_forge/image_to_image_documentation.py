from typing import List, Union
import numpy as np
from ..utils import (
from .base import Pipeline, build_pipeline_init_args

        Transform the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is used and
                the call may block forever.

        Return:
            An image (Image.Image) or a list of images (List["Image.Image"]) containing result(s). If the input is a
            single image, the return will be also a single image, if the input is a list of several images, it will
            return a list of transformed images.
        