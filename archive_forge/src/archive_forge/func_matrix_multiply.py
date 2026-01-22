import functools
import logging
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from io import DEFAULT_BUFFER_SIZE, BytesIO
from os import SEEK_CUR
from typing import (
from .errors import (
def matrix_multiply(a: TransformationMatrixType, b: TransformationMatrixType) -> TransformationMatrixType:
    return tuple((tuple((sum((float(i) * float(j) for i, j in zip(row, col))) for col in zip(*b))) for row in a))