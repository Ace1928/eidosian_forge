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
def read_union(decoder, writer_schema, named_schemas, reader_schema=None, options={}):
    index = decoder.read_index()
    idx_schema = writer_schema[index]
    if reader_schema:
        msg = f'schema mismatch: {writer_schema} not found in {reader_schema}'
        if not isinstance(reader_schema, list):
            if match_types(idx_schema, reader_schema, named_schemas):
                result = read_data(decoder, idx_schema, named_schemas, reader_schema, options)
            else:
                raise SchemaResolutionError(msg)
        else:
            for schema in reader_schema:
                if match_types(idx_schema, schema, named_schemas):
                    result = read_data(decoder, idx_schema, named_schemas, schema, options)
                    break
            else:
                raise SchemaResolutionError(msg)
    else:
        result = read_data(decoder, idx_schema, named_schemas, None, options)
    return_record_name_override = options.get('return_record_name_override')
    return_record_name = options.get('return_record_name')
    return_named_type_override = options.get('return_named_type_override')
    return_named_type = options.get('return_named_type')
    if return_named_type_override and is_single_name_union(writer_schema):
        return result
    elif return_named_type and extract_record_type(idx_schema) in NAMED_TYPES:
        return (idx_schema['name'], result)
    elif return_named_type and extract_record_type(idx_schema) not in AVRO_TYPES:
        return (named_schemas['writer'][idx_schema]['name'], result)
    elif return_record_name_override and is_single_record_union(writer_schema):
        return result
    elif return_record_name and extract_record_type(idx_schema) == 'record':
        return (idx_schema['name'], result)
    elif return_record_name and extract_record_type(idx_schema) not in AVRO_TYPES:
        return (named_schemas['writer'][idx_schema]['name'], result)
    else:
        return result