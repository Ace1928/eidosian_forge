import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
def make_serialized(schema, batches):
    with pa.BufferOutputStream() as sink:
        with pa.ipc.new_stream(sink, schema) as out:
            for batch in batches:
                out.write(batch)
        return sink.getvalue()