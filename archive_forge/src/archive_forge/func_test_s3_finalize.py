from datetime import datetime, timezone, timedelta
import gzip
import os
import pathlib
import subprocess
import sys
import pytest
import weakref
import pyarrow as pa
from pyarrow.tests.test_io import assert_file_not_found
from pyarrow.tests.util import (_filesystem_uri, ProxyHandler,
from pyarrow.fs import (FileType, FileInfo, FileSelector, FileSystem,
@pytest.mark.s3
def test_s3_finalize():
    code = 'if 1:\n        import pytest\n        from pyarrow.fs import (FileSystem, S3FileSystem,\n                                ensure_s3_initialized, finalize_s3)\n\n        fs, path = FileSystem.from_uri(\'s3://mf-nwp-models/README.txt\')\n        assert fs.region == \'eu-west-1\'\n        f = fs.open_input_stream(path)\n        f.read(50)\n\n        finalize_s3()\n\n        with pytest.raises(ValueError, match="S3 .* finalized"):\n            f.read(50)\n        with pytest.raises(ValueError, match="S3 .* finalized"):\n            fs.open_input_stream(path)\n        with pytest.raises(ValueError, match="S3 .* finalized"):\n            S3FileSystem(anonymous=True)\n        with pytest.raises(ValueError, match="S3 .* finalized"):\n            FileSystem.from_uri(\'s3://mf-nwp-models/README.txt\')\n        '
    subprocess.check_call([sys.executable, '-c', code])