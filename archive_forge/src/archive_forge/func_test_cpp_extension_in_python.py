import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
@pytest.mark.cython
def test_cpp_extension_in_python(tmpdir):
    from .test_cython import setup_template, compiler_opts, test_ld_path, test_util, here
    with tmpdir.as_cwd():
        pyx_file = 'extensions.pyx'
        shutil.copyfile(os.path.join(here, pyx_file), os.path.join(str(tmpdir), pyx_file))
        setup_code = setup_template.format(pyx_file=pyx_file, compiler_opts=compiler_opts, test_ld_path=test_ld_path)
        with open('setup.py', 'w') as f:
            f.write(setup_code)
        subprocess_env = test_util.get_modified_env_with_pythonpath()
        subprocess.check_call([sys.executable, 'setup.py', 'build_ext', '--inplace'], env=subprocess_env)
    sys.path.insert(0, str(tmpdir))
    mod = __import__('extensions')
    uuid_type = mod._make_uuid_type()
    assert uuid_type.extension_name == 'uuid'
    assert uuid_type.storage_type == pa.binary(16)
    array = mod._make_uuid_array()
    assert array.type == uuid_type
    assert array.to_pylist() == [b'abcdefghijklmno0', b'0onmlkjihgfedcba']
    assert array[0].as_py() == b'abcdefghijklmno0'
    assert array[1].as_py() == b'0onmlkjihgfedcba'
    buf = ipc_write_batch(pa.RecordBatch.from_arrays([array], ['uuid']))
    batch = ipc_read_batch(buf)
    reconstructed_array = batch.column(0)
    assert reconstructed_array.type == uuid_type
    assert reconstructed_array == array