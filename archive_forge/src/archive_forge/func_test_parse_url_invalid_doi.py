import os
import shutil
import time
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pytest
from ..utils import (
def test_parse_url_invalid_doi():
    """Should fail if we forget to not include // in the DOI link"""
    with pytest.raises(ValueError):
        parse_url('doi://XXX/XXX/fname.txt')