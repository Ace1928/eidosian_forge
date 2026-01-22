"""
Image Processing Module

This module provides functionalities for processing images, including format verification, contrast enhancement, resizing, hashing, compression, encryption, decryption, and decompression. It serves as a core component of the image interconversion GUI application, facilitating secure and efficient image manipulation.

Author: Lloyd Handyside
Creation Date: 2024-04-06
Last Modified: 2024-04-08

Functionalities:
- Ensure image format compatibility
- Enhance image contrast
- Resize images while maintaining aspect ratio
- Generate image hashes
- Compress and decompress image data
- Encrypt and decrypt image data
- Validate image properties
- Read image metadata
- Generate and verify checksums
- Parallel image resizing
"""

import io
import os
import asyncio
from PIL import Image, ImageEnhance, ExifTags
import hashlib
import zstd
from typing import Tuple, Dict, Any, Optional, List, Callable
from core_services import (
    ConfigManager,
    LoggingManager,
    EncryptionManager,
)  # Adjusted import for core_services
from concurrent.futures import ThreadPoolExecutor

# Initialize logging
LoggingManager.configure_logging(log_level="DEBUG")


class AppConfig:
    MAX_SIZE: Tuple[int, int] = (800, 600)  # Default values
    ENHANCEMENT_FACTOR: float = 1.5  # Default value


async def load_configurations():
    config_manager = ConfigManager()
    config_path = os.path.join(os.path.dirname(__file__), "config.ini")
    await config_manager.load_config(config_path, "ImageProcessing", file_type="ini")
    AppConfig.MAX_SIZE = tuple(
        map(
            int,
            (
                await config_manager.get(
                    "ImageProcessing", "MaxSize", fallback="800,600"
                )
            ).split(","),
        )
    )
    AppConfig.ENHANCEMENT_FACTOR = float(
        await config_manager.get("ImageProcessing", "EnhancementFactor", fallback="1.5")
    )


__all__ = [
    "ensure_image_format",
    "enhance_contrast",
    "resize_image",
    "get_image_hash",
    "compress",
    "encrypt",
    "decrypt",
    "decompress",
    "validate_image",
    "read_image_metadata",
    "ImageOperationError",
    "resize_images_parallel",
    "generate_checksum",
    "verify_checksum",
]


def log_function_call(func: Callable) -> Callable:
    """
    A decorator that logs the entry and exit of functions.
    """

    def wrapper(*args, **kwargs):
        LoggingManager.debug(f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        LoggingManager.debug(f"Exiting {func.__name__}")
        return result

    return wrapper


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
        LoggingManager.debug("Image format ensured successfully.")
        return image
    except Exception as e:
        LoggingManager.error(f"Failed to open image: {e}")
        raise ImageOperationError(f"Failed to open image: {e}")


@log_function_call
def enhance_contrast(
    image: Image.Image, enhancement_factor: float = AppConfig.ENHANCEMENT_FACTOR
) -> bytes:
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
            LoggingManager.debug("Image contrast enhanced successfully.")
            return output.getvalue()
    except Exception as e:
        LoggingManager.error(f"Error enhancing image contrast: {e}")
        raise ImageOperationError(f"Error enhancing image contrast: {e}")


@log_function_call
def resize_image(
    image: Image.Image, max_size: Tuple[int, int] = AppConfig.MAX_SIZE
) -> Image.Image:
    """
    Resizes an image to fit within a maximum size while maintaining aspect ratio.

    Args:
        image (Image.Image): The original image.
        max_size (Tuple[int, int], optional): A tuple of (max_width, max_height). Defaults to AppConfig.MAX_SIZE.

    Returns:
        Image.Image: The resized image.

    Raises:
        ImageOperationError: If resizing the image fails.
    """
    try:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        LoggingManager.debug(f"Image resized to max size {max_size}.")
        return image
    except Exception as e:
        LoggingManager.error(f"Error resizing image: {e}")
        raise ImageOperationError(f"Error resizing image: {e}")


@log_function_call
def get_image_hash(image_data: bytes) -> str:
    """
    Generates a SHA-512 hash of the image data.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        str: The hexadecimal hash of the image.
    """
    sha512_hash = hashlib.sha512()
    sha512_hash.update(image_data)
    LoggingManager.debug("Image hash successfully generated using SHA-512.")
    return sha512_hash.hexdigest()


@log_function_call
def compress(image_data: bytes, image_format: str) -> bytes:
    """
    Compresses image data along with its format and a checksum for integrity verification.

    Args:
        image_data (bytes): The raw image data.
        image_format (str): The format of the image.

    Returns:
        bytes: The compressed image data.
    """
    checksum = hashlib.sha256(image_data).hexdigest()
    formatted_data = f"{image_format}\x00{checksum}\x00".encode() + image_data
    compressed_data = zstd.compress(formatted_data)
    LoggingManager.debug("Image data compressed successfully.")
    return compressed_data


@log_function_call
def encrypt(data: bytes) -> bytes:
    """
    Encrypts data using the provided cipher suite.

    Args:
        data (bytes): The data to encrypt.

    Returns:
        bytes: The encrypted data.
    """
    encrypted_data = EncryptionManager.encrypt(data)
    LoggingManager.debug("Data encrypted successfully.")
    return encrypted_data


@log_function_call
def decrypt(encrypted_data: bytes) -> bytes:
    """
    Decrypts data using the provided cipher suite.

    Args:
        encrypted_data (bytes): The encrypted data.

    Returns:
        bytes: The decrypted data.
    """
    decrypted_data = EncryptionManager.decrypt(encrypted_data)
    LoggingManager.debug("Data decrypted successfully.")
    return decrypted_data


@log_function_call
def decompress(data: bytes) -> Tuple[bytes, str]:
    """
    Decompresses data and extracts the image format and raw image data.

    Args:
        data (bytes): The compressed data.

    Returns:
        Tuple[bytes, str]: The raw image data and its format.
    """
    decompressed_data = zstd.decompress(data)
    image_format, checksum, image_data = decompressed_data.split(b"\x00", 2)
    LoggingManager.debug("Data decompressed successfully.")
    return image_data, image_format.decode()


@log_function_call
def generate_checksum(image_data: bytes) -> str:
    """
    Generates a SHA-256 checksum for the given image data.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        str: The hexadecimal checksum of the image.
    """
    checksum = hashlib.sha256(image_data).hexdigest()
    LoggingManager.debug("Checksum generated successfully.")
    return checksum


@log_function_call
def verify_checksum(image_data: bytes, expected_checksum: str) -> bool:
    """
    Verifies the integrity of the image data against the expected checksum.

    Args:
        image_data (bytes): The raw image data.
        expected_checksum (str): The expected checksum for verification.

    Returns:
        bool: True if the checksum matches, False otherwise.
    """
    actual_checksum = hashlib.sha256(image_data).hexdigest()
    is_valid = actual_checksum == expected_checksum
    LoggingManager.debug(f"Checksum verification result: {is_valid}.")
    return is_valid


@log_function_call
def validate_image(image: Image.Image) -> bool:
    """
    Validates the image format and size.

    Args:
        image (Image.Image): The image to validate.

    Returns:
        bool: True if the image is valid, False otherwise.

    Raises:
        ImageOperationError: If image validation fails.
    """
    try:
        valid_formats = ["JPEG", "PNG", "BMP", "GIF"]
        is_valid = (
            image.format in valid_formats
            and image.width <= 4000
            and image.height <= 4000
        )
        LoggingManager.debug(f"Image validation result: {is_valid}.")
        return is_valid
    except Exception as e:
        LoggingManager.error(f"Error validating image: {e}")
        raise ImageOperationError(f"Error validating image: {e}")


@log_function_call
def read_image_metadata(image_data: bytes) -> Dict[str, Any]:
    """
    Reads EXIF metadata from an image.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        Dict[str, Any]: A dictionary containing EXIF metadata, if available.

    Raises:
        ImageOperationError: If reading metadata fails.
    """
    try:
        with Image.open(io.BytesIO(image_data)) as image:
            exif_data = {}
            if hasattr(image, "_getexif"):
                exif = image._getexif()
                if exif is not None:
                    for tag, value in exif.items():
                        decoded = ExifTags.TAGS.get(tag, tag)
                        exif_data[decoded] = value
            LoggingManager.debug("Image metadata read successfully.")
            return exif_data
    except Exception as e:
        LoggingManager.error(f"Error reading image metadata: {e}")
        raise ImageOperationError(f"Error reading image metadata: {e}")


@log_function_call
def resize_images_parallel(
    images: List[bytes], max_size: Tuple[int, int] = AppConfig.MAX_SIZE
) -> List[Image.Image]:
    """
    Resizes a list of images in parallel.

    Args:
        images (List[bytes]): The list of raw image data.
        max_size (Tuple[int, int], optional): The maximum width and height. Defaults to AppConfig.MAX_SIZE.

    Returns:
        List[Image.Image]: The list of resized images.
    """

    def process_image(image_data):
        try:
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            LoggingManager.error(f"Error processing image: {e}")
            raise ImageOperationError(f"Error processing image: {e}")

    with ThreadPoolExecutor() as executor:
        resized_images = list(executor.map(process_image, images))
    return resized_images


def get_cipher_suite() -> str:
    """
    Retrieves the cipher suite using the encryption key from an environment variable.

    Returns:
        str: The cipher suite.
    """
    return EncryptionManager.get_cipher_suite()


# Define a base class for plugins
class ImageProcessingPlugin:
    def process(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError


# Example plugin
class SepiaTonePlugin(ImageProcessingPlugin):
    def process(self, image: Image.Image) -> Image.Image:
        # Apply sepia tone transformation
        return image  # Transformed image


# Register and use plugins
plugins = [SepiaTonePlugin()]


def apply_plugins(image: Image.Image) -> Image.Image:
    for plugin in plugins:
        image = plugin.process(image)
    return image


# TODO:
# - Implement support for additional image formats.
# - Optimize performance for large image files.
# - Enhance encryption mechanisms for increased security.
# - Improve parallel processing implementation for better scalability.

# Known Issues:
# - Compression may result in quality loss for certain image formats.
# - Parallel processing implementation may need optimization for large batches of images.
