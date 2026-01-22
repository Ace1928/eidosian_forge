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
def merge_translated_page(self, page2: 'PageObject', tx: float, ty: float, over: bool=True, expand: bool=False) -> None:
    """
        mergeTranslatedPage is similar to merge_page, but the stream to be
        merged is translated by applying a transformation matrix.

        Args:
          page2: the page to be merged into this one.
          tx: The translation on X axis
          ty: The translation on Y axis
          over: set the page2 content over page1 if True(default) else under
          expand: Whether the page should be expanded to fit the
            dimensions of the page to be merged.
        """
    op = Transformation().translate(tx, ty)
    self.merge_transformed_page(page2, op, over, expand)