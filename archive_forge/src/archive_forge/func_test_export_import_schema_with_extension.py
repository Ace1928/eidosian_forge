import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
@needs_cffi
def test_export_import_schema_with_extension():
    check_export_import_schema(make_extension_schema, make_extension_storage_schema)
    with registered_extension_type(ParamExtType(1)):
        check_export_import_schema(make_extension_schema)