import io
import functools
import logging
import time
import warnings
import smart_open.bytebuffer
import smart_open.concurrency
import smart_open.utils
from smart_open import constants
def to_boto3(self, resource):
    """Create an **independent** `boto3.s3.Object` instance that points to
        the same S3 object as this instance.
        Changes to the returned object will not affect the current instance.
        """
    assert resource, 'resource must be a boto3.resource instance'
    return resource.Object(self._bucket, self._key)