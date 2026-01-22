import shlex
import subprocess
import time
import uuid
import pytest
from pandas.compat import (
import pandas.util._test_decorators as td
import pandas.io.common as icom
from pandas.io.parsers import read_csv
@pytest.fixture
def s3_private_bucket(s3_resource):
    bucket = s3_resource.Bucket(f'cant_get_it-{uuid.uuid4()}')
    bucket.create(ACL='private')
    yield bucket
    bucket.objects.delete()
    bucket.delete()