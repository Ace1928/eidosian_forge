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
def temp_blobs(temp_bucket):
    """Yields two blobs that are deleted after the test completes."""
    bucket = temp_bucket
    accessible_blob = bucket.blob(_ACCESSIBLE_OBJECT_NAME)
    accessible_blob.upload_from_string(_ACCESSIBLE_CONTENT)
    inaccessible_blob = bucket.blob(_INACCESSIBLE_OBJECT_NAME)
    inaccessible_blob.upload_from_string(_INACCESSIBLE_CONTENT)
    yield (accessible_blob, inaccessible_blob)
    bucket.delete_blobs([accessible_blob, inaccessible_blob])