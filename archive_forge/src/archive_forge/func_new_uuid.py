import functools
import os
import time
import uuid
from testtools import testcase
def new_uuid():
    """Return a string UUID."""
    return uuid.uuid4().hex