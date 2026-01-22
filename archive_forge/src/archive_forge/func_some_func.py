from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
@api_versions.wraps('2.2', '2.6')
def some_func(*args, **kwargs):
    checker(*args, **kwargs)