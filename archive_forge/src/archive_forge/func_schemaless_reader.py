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
def schemaless_reader(fo: IO, writer_schema: Schema, reader_schema: Optional[Schema]=None, return_record_name: bool=False, return_record_name_override: bool=False, handle_unicode_errors: str='strict', return_named_type: bool=False, return_named_type_override: bool=False) -> AvroMessage:
    """Reads a single record written using the
    :meth:`~fastavro._write_py.schemaless_writer`

    Parameters
    ----------
    fo
        Input stream
    writer_schema
        Schema used when calling schemaless_writer
    reader_schema
        If the schema has changed since being written then the new schema can
        be given to allow for schema migration
    return_record_name
        If true, when reading a union of records, the result will be a tuple
        where the first value is the name of the record and the second value is
        the record itself
    return_record_name_override
        If true, this will modify the behavior of return_record_name so that
        the record name is only returned for unions where there is more than
        one record. For unions that only have one record, this option will make
        it so that the record is returned by itself, not a tuple with the name.
    return_named_type
        If true, when reading a union of named types, the result will be a tuple
        where the first value is the name of the type and the second value is
        the record itself
        NOTE: Using this option will ignore return_record_name and
        return_record_name_override
    return_named_type_override
        If true, this will modify the behavior of return_named_type so that
        the named type is only returned for unions where there is more than
        one named type. For unions that only have one named type, this option
        will make it so that the named type is returned by itself, not a tuple
        with the name
    handle_unicode_errors
        Default `strict`. Should be set to a valid string that can be used in
        the errors argument of the string decode() function. Examples include
        `replace` and `ignore`


    Example::

        parsed_schema = fastavro.parse_schema(schema)
        with open('file', 'rb') as fp:
            record = fastavro.schemaless_reader(fp, parsed_schema)

    Note: The ``schemaless_reader`` can only read a single record.
    """
    if writer_schema == reader_schema:
        reader_schema = None
    named_schemas: Dict[str, NamedSchemas] = _default_named_schemas()
    writer_schema = parse_schema(writer_schema, named_schemas['writer'])
    if reader_schema:
        reader_schema = parse_schema(reader_schema, named_schemas['reader'])
    decoder = BinaryDecoder(fo)
    options = {'return_record_name': return_record_name, 'return_record_name_override': return_record_name_override, 'handle_unicode_errors': handle_unicode_errors, 'return_named_type': return_named_type, 'return_named_type_override': return_named_type_override}
    return read_data(decoder, writer_schema, named_schemas, reader_schema, options)