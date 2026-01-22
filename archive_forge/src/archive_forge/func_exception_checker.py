from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
def exception_checker(exc):
    return isinstance(exc, ValueError) and exc.args[0] < 5