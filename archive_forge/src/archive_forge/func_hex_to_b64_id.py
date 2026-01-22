import base64
import hashlib
import mmap
import os
import sys
from pathlib import Path
from typing import NewType, Union
from wandb.sdk.lib.paths import StrPath
def hex_to_b64_id(encoded_string: Union[str, bytes]) -> B64MD5:
    if isinstance(encoded_string, bytes):
        encoded_string = encoded_string.decode('utf-8')
    as_str = bytes.fromhex(encoded_string)
    return B64MD5(base64.standard_b64encode(as_str).decode('utf-8'))