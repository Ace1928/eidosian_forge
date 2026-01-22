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

        Synchronously wraps the asynchronous test for encryption and decryption.

        This test method provides a synchronous interface to the asynchronous test_encrypt_and_decrypt method,
        allowing it to be executed as part of the synchronous unit test suite. It utilizes the asyncio.run function to
        execute the asynchronous test method.
        