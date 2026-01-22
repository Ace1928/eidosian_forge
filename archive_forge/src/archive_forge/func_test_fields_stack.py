from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def test_fields_stack(gdb_arrow):
    check_stack_repr(gdb_arrow, 'int_field', 'arrow::field("ints", arrow::int64())')
    check_stack_repr(gdb_arrow, 'float_field', 'arrow::field("floats", arrow::float32(), nullable=false)')