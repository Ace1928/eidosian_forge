import unittest
import io
import os
import asyncio
from PIL import Image, ImageEnhance, ExifTags
import hashlib
import zstd
from typing import Tuple, Dict, Any, Optional, List, Callable
from core_services import (
from concurrent.futures import ThreadPoolExecutor
from cryptography.fernet import Fernet
import traceback
@log_function_call
def enhance_contrast(image: Image.Image, enhancement_factor: float=AppConfig.ENHANCEMENT_FACTOR) -> bytes:
    """
    Enhances the contrast of an image.

    Args:
        image (Image.Image): The image to enhance.
        enhancement_factor (float, optional): The factor by which to enhance the image's contrast. Defaults to AppConfig.ENHANCEMENT_FACTOR.

    Returns:
        bytes: The enhanced image data.

    Raises:
        ImageOperationError: If contrast enhancement fails.
    """
    try:
        enhancer: ImageEnhance.Contrast = ImageEnhance.Contrast(image)
        enhanced_image: Image.Image = enhancer.enhance(enhancement_factor)
        with io.BytesIO() as output:
            enhanced_image.save(output, format=image.format)
            LoggingManager.debug('Image contrast enhanced successfully.')
            return output.getvalue()
    except Exception as e:
        LoggingManager.error(f'Error enhancing image contrast: {e}')
        raise ImageOperationError(f'Error enhancing image contrast due to: {str(e)}') from e