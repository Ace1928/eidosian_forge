from __future__ import annotations
import codecs
import os
import pathlib
import sys
from io import BufferedIOBase, BytesIO, RawIOBase, StringIO, TextIOBase, TextIOWrapper
from typing import (
from urllib.parse import urljoin
from urllib.request import Request, url2pathname
from xml.sax import xmlreader
import rdflib.util
from rdflib import __version__
from rdflib._networking import _urlopen
from rdflib.namespace import Namespace
from rdflib.term import URIRef
def create_input_source(source: Optional[Union[IO[bytes], TextIO, InputSource, str, bytes, pathlib.PurePath]]=None, publicID: Optional[str]=None, location: Optional[str]=None, file: Optional[Union[BinaryIO, TextIO]]=None, data: Optional[Union[str, bytes, dict]]=None, format: Optional[str]=None) -> InputSource:
    """
    Return an appropriate InputSource instance for the given
    parameters.
    """
    non_empty_arguments = list(filter(lambda v: v is not None, [source, location, file, data]))
    if len(non_empty_arguments) != 1:
        raise ValueError('exactly one of source, location, file or data must be given')
    input_source = None
    if source is not None:
        if TYPE_CHECKING:
            assert file is None
            assert data is None
            assert location is None
        if isinstance(source, InputSource):
            input_source = source
        elif isinstance(source, str):
            location = source
        elif isinstance(source, pathlib.PurePath):
            location = str(source)
        elif isinstance(source, bytes):
            data = source
        elif hasattr(source, 'read') and (not isinstance(source, Namespace)):
            f = source
            input_source = InputSource()
            if hasattr(source, 'encoding'):
                input_source.setCharacterStream(source)
                input_source.setEncoding(source.encoding)
                try:
                    b = source.buffer
                    input_source.setByteStream(b)
                except (AttributeError, LookupError):
                    input_source.setByteStream(source)
            else:
                input_source.setByteStream(f)
            if f is sys.stdin:
                input_source.setSystemId('file:///dev/stdin')
            elif hasattr(f, 'name'):
                input_source.setSystemId(f.name)
        else:
            raise Exception("Unexpected type '%s' for source '%s'" % (type(source), source))
    absolute_location = None
    auto_close = False
    if location is not None:
        if TYPE_CHECKING:
            assert file is None
            assert data is None
            assert source is None
        absolute_location, auto_close, file, input_source = _create_input_source_from_location(file=file, format=format, input_source=input_source, location=location)
    if file is not None:
        if TYPE_CHECKING:
            assert location is None
            assert data is None
            assert source is None
        input_source = FileInputSource(file)
    if data is not None:
        if TYPE_CHECKING:
            assert location is None
            assert file is None
            assert source is None
        if isinstance(data, dict):
            input_source = PythonInputSource(data)
            auto_close = True
        elif isinstance(data, (str, bytes, bytearray)):
            input_source = StringInputSource(data)
            auto_close = True
        else:
            raise RuntimeError(f'parse data can only str, or bytes. not: {type(data)}')
    if input_source is None:
        raise Exception('could not create InputSource')
    else:
        input_source.auto_close |= auto_close
        if publicID is not None:
            input_source.setPublicId(publicID)
        elif input_source.getPublicId() is None:
            input_source.setPublicId(absolute_location or '')
        return input_source