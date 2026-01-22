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
def get_image_hash(image_data: bytes) -> str:
    """
    Generates a SHA-512 hash of the image data.

    This function takes raw image data as input and computes its SHA-512 hash. The hash is then returned as a hexadecimal
    string. This process is crucial for ensuring data integrity and uniqueness across image processing operations.

    Args:
        image_data (bytes): The raw image data to be hashed.

    Returns:
        str: The hexadecimal representation of the SHA-512 hash of the image data.

    Raises:
        ImageOperationError: If there's an error during the hashing process.
    """
    try:
        sha512_hash = hashlib.sha512()
        sha512_hash.update(image_data)
        LoggingManager.debug('Image hash successfully generated using SHA-512.')
        return sha512_hash.hexdigest()
    except Exception as e:
        LoggingManager.error(f'Error generating image hash: {e}')
        raise ImageOperationError(f'Error generating image hash due to: {str(e)}') from e