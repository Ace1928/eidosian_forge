import bz2
import json
import lzma
import zlib
from datetime import datetime, timezone
from decimal import Context
from io import BytesIO
from struct import error as StructError
from typing import IO, Union, Optional, Generic, TypeVar, Iterator, Dict
from warnings import warn
from .io.binary_decoder import BinaryDecoder
from .io.json_decoder import AvroJSONDecoder
from .logical_readers import LOGICAL_READERS
from .schema import (
from .types import Schema, AvroMessage, NamedSchemas
from ._read_common import (
from .const import NAMED_TYPES, AVRO_TYPES
def read_map(decoder, writer_schema, named_schemas, reader_schema=None, options={}):
    if reader_schema:

        def item_reader(decoder, w_schema, r_schema):
            return read_data(decoder, w_schema['values'], named_schemas, r_schema['values'], options)
    else:

        def item_reader(decoder, w_schema, r_schema):
            return read_data(decoder, w_schema['values'], named_schemas, None, options)
    read_items = {}
    decoder.read_map_start()
    for item in decoder.iter_map():
        key = decoder.read_utf8()
        read_items[key] = item_reader(decoder, writer_schema, reader_schema)
    decoder.read_map_end()
    return read_items