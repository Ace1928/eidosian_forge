from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
@pytest.fixture(scope='session')
def gdb_arrow(gdb):
    if 'deb' not in pa.cpp_build_info.build_type:
        pytest.skip('Arrow C++ debug symbols not available')
    skip_if_gdb_script_unavailable()
    gdb.run_command(f'source {gdb_script}')
    lib_path_var = 'PATH' if sys.platform == 'win32' else 'LD_LIBRARY_PATH'
    lib_path = os.environ.get(lib_path_var)
    if lib_path:
        gdb.run_command(f'set env {lib_path_var} {lib_path}')
    code = 'from pyarrow.lib import _gdb_test_session; _gdb_test_session()'
    out = gdb.run_command(f"run -c '{code}'")
    assert 'Trace/breakpoint trap' in out or 'received signal' in out, out
    gdb.select_frame('arrow::gdb::TestSession')
    return gdb