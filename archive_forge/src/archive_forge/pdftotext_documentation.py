from __future__ import annotations
import os
import asyncio
import subprocess
from pathlib import Path
from ..base import (
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar

        Convert the source to the target

        source: /path/to/file.pdf
        target: '.txt'
        