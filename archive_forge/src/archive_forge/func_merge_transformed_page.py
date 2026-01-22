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
def merge_transformed_page(self, page2: 'PageObject', ctm: Union[CompressedTransformationMatrix, Transformation], over: bool=True, expand: bool=False) -> None:
    """
        merge_transformed_page is similar to merge_page, but a transformation
        matrix is applied to the merged stream.

        Args:
          page2: The page to be merged into this one.
          ctm: a 6-element tuple containing the operands of the
                 transformation matrix
          over: set the page2 content over page1 if True(default) else under
          expand: Whether the page should be expanded to fit the dimensions
            of the page to be merged.
        """
    if isinstance(ctm, Transformation):
        ctm = ctm.ctm
    self._merge_page(page2, lambda page2Content: PageObject._add_transformation_matrix(page2Content, page2.pdf, cast(CompressedTransformationMatrix, ctm)), ctm, over, expand)