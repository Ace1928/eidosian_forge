import sys
import warnings
from pathlib import Path
from typing import Tuple
import requests
from wasabi import Printer, msg
from .. import about
from ..util import (
from ._util import app
def reformat_version(version: str) -> str:
    """Hack to reformat old versions ending on '-alpha' to match pip format."""
    if version.endswith('-alpha'):
        return version.replace('-alpha', 'a0')
    return version.replace('-alpha', 'a')