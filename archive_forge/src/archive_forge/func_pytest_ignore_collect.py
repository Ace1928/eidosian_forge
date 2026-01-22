import os
import sys
import pytest
import six
def pytest_ignore_collect(path, config):
    """Skip App Engine tests in python 3 or if no SDK is available."""
    if 'appengine' in str(path):
        if not six.PY2:
            return True
        if not os.environ.get('GAE_SDK_PATH'):
            return True
    return False