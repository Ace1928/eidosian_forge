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
def s3_resource(s3_base):
    import boto3
    s3 = boto3.resource('s3', endpoint_url=s3_base)
    return s3