from __future__ import annotations
import os
import asyncio
import subprocess
from pathlib import Path
from ..base import (
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar
def validate_enabled(self, **kwargs) -> bool:
    """
        Validate whether the converter is enabled
        """
    from shutil import which
    return which('pdftotext') is not None