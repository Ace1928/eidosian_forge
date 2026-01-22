import re
import uuid
import google.auth
from google.auth import downscoped
from google.auth.transport import requests
from google.cloud import exceptions
from google.cloud import storage
from google.oauth2 import credentials
import pytest
@pytest.fixture(scope='module')
def temp_bucket():
    """Yields a bucket that is deleted after the test completes."""
    bucket = None
    while bucket is None or bucket.exists():
        bucket_name = 'auth-python-downscope-test-{}'.format(uuid.uuid4())
        bucket = storage.Client().bucket(bucket_name)
    bucket = storage.Client().create_bucket(bucket.name)
    yield bucket
    bucket.delete(force=True)