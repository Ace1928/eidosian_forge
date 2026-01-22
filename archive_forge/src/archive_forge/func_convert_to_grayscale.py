from typing import Dict, Optional, Union
import numpy as np
from ... import is_vision_available
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, logging, requires_backends
def convert_to_grayscale(image: ImageInput, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> ImageInput:
    """
    Converts an image to grayscale format using the NTSC formula. Only support numpy and PIL Image. TODO support torch
    and tensorflow grayscale conversion

    This function is supposed to return a 1-channel image, but it returns a 3-channel image with the same value in each
    channel, because of an issue that is discussed in :
    https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

    Args:
        image (Image):
            The image to convert.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image.
    """
    requires_backends(convert_to_grayscale, ['vision'])
    if isinstance(image, np.ndarray):
        if input_data_format == ChannelDimension.FIRST:
            gray_image = image[0, ...] * 0.2989 + image[1, ...] * 0.587 + image[2, ...] * 0.114
            gray_image = np.stack([gray_image] * 3, axis=0)
        elif input_data_format == ChannelDimension.LAST:
            gray_image = image[..., 0] * 0.2989 + image[..., 1] * 0.587 + image[..., 2] * 0.114
            gray_image = np.stack([gray_image] * 3, axis=-1)
        return gray_image
    if not isinstance(image, PIL.Image.Image):
        return image
    image = image.convert('L')
    return image