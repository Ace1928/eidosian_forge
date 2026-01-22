import os
import platform
from functools import wraps
from pathlib import PurePath, PurePosixPath
from typing import Any, NewType, Union
Act like a PurePosixPath for the / operator, but return a LogicalPath.