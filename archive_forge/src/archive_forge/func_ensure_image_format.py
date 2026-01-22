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
def ensure_image_format(image_data: bytes) -> Image.Image:
    """
    Ensures the given image data can be opened and returns the Image object.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        Image.Image: The opened image.

    Raises:
        ImageOperationError: If the image cannot be opened.
    """
    try:
        image: Image.Image = Image.open(io.BytesIO(image_data))
        LoggingManager.debug('Image format ensured successfully.')
        return image
    except Exception as e:
        LoggingManager.error(f'Failed to open image: {e}')
        raise ImageOperationError(f'Failed to open image due to: {str(e)}') from e