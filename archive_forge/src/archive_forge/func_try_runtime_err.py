import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def try_runtime_err():
    with ignore_assertion_error:
        raise RuntimeError