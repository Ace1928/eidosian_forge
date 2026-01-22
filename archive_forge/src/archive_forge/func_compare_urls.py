import logging
import os
import shutil
import sys
import urllib.parse
from typing import (
from pip._internal.cli.spinners import SpinnerInterface
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import (
from pip._internal.utils.subprocess import (
from pip._internal.utils.urls import get_url_scheme
@classmethod
def compare_urls(cls, url1: str, url2: str) -> bool:
    """
        Compare two repo URLs for identity, ignoring incidental differences.
        """
    return cls.normalize_url(url1) == cls.normalize_url(url2)