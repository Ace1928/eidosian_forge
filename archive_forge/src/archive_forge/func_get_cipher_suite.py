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
def get_cipher_suite() -> str:
    """
    Retrieves the cipher suite using the encryption key from an environment variable.

    This function encapsulates the logic to fetch the cipher suite designated for encryption or decryption processes.
    It leverages the EncryptionManager, a centralized entity responsible for managing encryption keys and related
    configurations, to obtain the cipher suite. The retrieval process is designed to ensure that the encryption
    mechanism remains consistent and secure across the application.

    Returns:
        str: The cipher suite identifier, which is essential for initializing the encryption or decryption process.

    Raises:
        EncryptionError: If the cipher suite cannot be retrieved due to misconfiguration or absence of the encryption key.
    """
    try:
        cipher_suite = EncryptionManager.get_cipher_suite()
        LoggingManager.debug(f'Retrieved cipher suite: {cipher_suite}')
        return cipher_suite
    except Exception as e:
        LoggingManager.error(f'Failed to retrieve cipher suite: {e}')
        raise EncryptionError(f'Failed to retrieve cipher suite due to: {str(e)}') from e