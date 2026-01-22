from __future__ import absolute_import, division, print_function
import sys
import math
from collections import OrderedDict
from datetime import datetime, date, time
from decimal import Decimal
from petl.compat import izip_longest, text_type, string_types, PY3
from petl.io.sources import read_source_from_arg, write_source_from_arg
from petl.transform.headers import skip, setheader
from petl.util.base import Table, dicts, fieldnames, iterpeek, wrap
def toavro(table, target, schema=None, sample=9, codec='deflate', compression_level=None, **avro_args):
    """
    Write the table into a new avro file according to schema passed.

    This method assume that each column has values with the same type 
    for all rows of the source `table`.

    `Apache Avro`_ is a data
    serialization framework. It is used in data serialization (especially in
    Hadoop ecosystem), for dataexchange for databases (Redshift) and RPC 
    protocols (like in Kafka). It has libraries to support many languages and
    generally is faster and safer than text formats like Json, XML or CSV.

    The `target` argument is the file path for creating the avro file.
    Note that if a file already exists at the given location, it will be
    overwritten.

    The `schema` argument (dict) defines the rows field structure of the file.
    Check fastavro `documentation`_ and Avro schema `reference`_ for details.

    The `sample` argument (int, optional) defines how many rows are inspected
    for discovering the field types and building a schema for the avro file 
    when the `schema` argument is not passed.

    The `codec` argument (string, optional) sets the compression codec used to
    shrink data in the file. It can be 'null', 'deflate' (default), 'bzip2' or
    'snappy', 'zstandard', 'lz4', 'xz' (if installed)

    The `compression_level` argument (int, optional) sets the level of 
    compression to use with the specified codec (if the codec supports it)

    Additionally there are support for passing extra options in the 
    argument `**avro_args` that are fowarded directly to fastavro. Check the
    fastavro `documentation`_ for reference.

    The avro file format preserves type information, i.e., reading and writing
    is round-trippable for tables with non-string data values. However the
    conversion from Python value types to avro fields is not perfect. Use the
    `schema` argument to define proper type to the conversion.

    The following avro types are supported by the schema: null, boolean, 
    string, int, long, float, double, bytes, fixed, enum, 
    :ref:`array <array_schema>`, map, union, record, and recursive types 
    defined in :ref:`complex schemas <complex_schema>`.

    Also :ref:`logical types <logical_schema>` are supported and translated to 
    coresponding python types: long timestamp-millis, long timestamp-micros, int date, 
    bytes decimal, fixed decimal, string uuid, int time-millis, long time-micros.

    Example usage for writing files::

        >>> # set up a Avro file to demonstrate with
        >>> table2 = [['name', 'friends', 'age'],
        ...           ['Bob', 42, 33],
        ...           ['Jim', 13, 69],
        ...           ['Joe', 86, 17],
        ...           ['Ted', 23, 51]]
        ...
        >>> schema2 = {
        ...     'doc': 'Some people records.',
        ...     'name': 'People',
        ...     'namespace': 'test',
        ...     'type': 'record',
        ...     'fields': [
        ...         {'name': 'name', 'type': 'string'},
        ...         {'name': 'friends', 'type': 'int'},
        ...         {'name': 'age', 'type': 'int'},
        ...     ]
        ... }
        ...
        >>> # now demonstrate what writing with toavro()
        >>> import petl as etl
        >>> etl.toavro(table2, 'example.file2.avro', schema=schema2)
        ...
        >>> # this was what was saved above
        >>> tbl2 = etl.fromavro('example.file2.avro')
        >>> tbl2
        +-------+---------+-----+
        | name  | friends | age |
        +=======+=========+=====+
        | 'Bob' |      42 |  33 |
        +-------+---------+-----+
        | 'Jim' |      13 |  69 |
        +-------+---------+-----+
        | 'Joe' |      86 |  17 |
        +-------+---------+-----+
        | 'Ted' |      23 |  51 |
        +-------+---------+-----+

    .. versionadded:: 1.4.0

    .. _Apache Avro: https://avro.apache.org/docs/current/spec.html
    .. _reference: https://avro.apache.org/docs/current/spec.html#schemas
    .. _documentation : https://fastavro.readthedocs.io/en/latest/writer.html

    """
    _write_toavro(table, target=target, mode='wb', schema=schema, sample=sample, codec=codec, compression_level=compression_level, **avro_args)