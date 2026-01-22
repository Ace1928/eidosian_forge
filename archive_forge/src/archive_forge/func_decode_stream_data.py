import math
import struct
import zlib
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ._utils import (
from .constants import CcittFaxDecodeParameters as CCITT
from .constants import ColorSpaces
from .constants import FilterTypeAbbreviations as FTA
from .constants import FilterTypes as FT
from .constants import ImageAttributes as IA
from .constants import LzwFilterParameters as LZW
from .constants import StreamAttributes as SA
from .errors import DeprecationError, PdfReadError, PdfStreamError
from .generic import (
def decode_stream_data(stream: Any) -> Union[bytes, str]:
    """
    Decode the stream data based on the specified filters.

    This function decodes the stream data using the filters provided in the
    stream. It supports various filter types, including FlateDecode,
    ASCIIHexDecode, RunLengthDecode, LZWDecode, ASCII85Decode, DCTDecode, JPXDecode, and
    CCITTFaxDecode.

    Args:
        stream: The input stream object containing the data and filters.

    Returns:
        The decoded stream data.

    Raises:
        NotImplementedError: If an unsupported filter type is encountered.
    """
    filters = stream.get(SA.FILTER, ())
    if isinstance(filters, IndirectObject):
        filters = cast(ArrayObject, filters.get_object())
    if not isinstance(filters, ArrayObject):
        filters = (filters,)
    decodparms = stream.get(SA.DECODE_PARMS, ({},) * len(filters))
    if not isinstance(decodparms, (list, tuple)):
        decodparms = (decodparms,)
    data: bytes = b_(stream._data)
    if data:
        for filter_type, params in zip(filters, decodparms):
            if isinstance(params, NullObject):
                params = {}
            if filter_type in (FT.FLATE_DECODE, FTA.FL):
                data = FlateDecode.decode(data, params)
            elif filter_type in (FT.ASCII_HEX_DECODE, FTA.AHx):
                data = ASCIIHexDecode.decode(data)
            elif filter_type in (FT.RUN_LENGTH_DECODE, FTA.RL):
                data = RunLengthDecode.decode(data)
            elif filter_type in (FT.LZW_DECODE, FTA.LZW):
                data = LZWDecode.decode(data, params)
            elif filter_type in (FT.ASCII_85_DECODE, FTA.A85):
                data = ASCII85Decode.decode(data)
            elif filter_type == FT.DCT_DECODE:
                data = DCTDecode.decode(data)
            elif filter_type == FT.JPX_DECODE:
                data = JPXDecode.decode(data)
            elif filter_type == FT.CCITT_FAX_DECODE:
                height = stream.get(IA.HEIGHT, ())
                data = CCITTFaxDecode.decode(data, params, height)
            elif filter_type == '/Crypt':
                if '/Name' not in params and '/Type' not in params:
                    pass
                else:
                    raise NotImplementedError('/Crypt filter with /Name or /Type not supported yet')
            else:
                raise NotImplementedError(f'unsupported filter {filter_type}')
    return data