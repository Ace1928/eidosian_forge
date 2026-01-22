import base64
import hashlib
import mmap
import os
import sys
from pathlib import Path
from typing import NewType, Union
from wandb.sdk.lib.paths import StrPath
def md5_file_hex(*paths: StrPath) -> HexMD5:
    return HexMD5(_md5_file_hasher(*paths).hexdigest())