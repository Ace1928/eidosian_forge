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
def skip_data(decoder, writer_schema, named_schemas):
    record_type = extract_record_type(writer_schema)
    reader_fn = SKIPS.get(record_type)
    if reader_fn:
        reader_fn(decoder, writer_schema, named_schemas)
    else:
        skip_data(decoder, named_schemas['writer'][record_type], named_schemas)