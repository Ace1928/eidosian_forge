import math
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import (
from ._cmap import build_char_map, unknown_char_map
from ._protocols import PdfCommonDocProtocol
from ._text_extraction import (
from ._utils import (
from .constants import AnnotationDictionaryAttributes as ADA
from .constants import ImageAttributes as IA
from .constants import PageAttributes as PG
from .constants import Resources as RES
from .errors import PageSizeNotDefinedError, PdfReadError
from .filters import _xobj_to_image
from .generic import (
def scale_by(self, factor: float) -> None:
    """
        Scale a page by the given factor by applying a transformation matrix to
        its content and updating the page size.

        Args:
            factor: The scaling factor (for both X and Y axis).
        """
    self.scale(factor, factor)