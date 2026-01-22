import logging
import re
import sys
from io import BytesIO
from typing import (
from .._protocols import PdfReaderProtocol, PdfWriterProtocol, XmpInformationProtocol
from .._utils import (
from ..constants import (
from ..constants import FilterTypes as FT
from ..constants import StreamAttributes as SA
from ..constants import TypArguments as TA
from ..constants import TypFitArguments as TF
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfReadError, PdfStreamError
from ._base import (
from ._fit import Fit
from ._utils import read_hex_string_from_stream, read_string_from_stream
@property
def xmp_metadata(self) -> Optional[XmpInformationProtocol]:
    """
        Retrieve XMP (Extensible Metadata Platform) data relevant to the this
        object, if available.

        See Table 347 — Additional entries in a metadata stream dictionary.

        Returns:
          Returns a :class:`~pypdf.xmp.XmpInformation` instance
          that can be used to access XMP metadata from the document.  Can also
          return None if no metadata was found on the document root.
        """
    from ..xmp import XmpInformation
    metadata = self.get('/Metadata', None)
    if metadata is None:
        return None
    metadata = metadata.get_object()
    if not isinstance(metadata, XmpInformation):
        metadata = XmpInformation(metadata)
        self[NameObject('/Metadata')] = metadata
    return metadata