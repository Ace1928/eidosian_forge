import re
import shutil
import subprocess
import warnings
from io import BytesIO, TextIOWrapper
from nbconvert.utils.version import check_version
from .exceptions import ConversionException
def clean_cache():
    """Clean the internal cache."""
    global __version
    __version = None