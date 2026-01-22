import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
@excutils.exception_filter
@staticmethod
def ignore_exceptions(ex):
    """Ignore some exceptions S."""
    return isinstance(ex, ignore_classes)