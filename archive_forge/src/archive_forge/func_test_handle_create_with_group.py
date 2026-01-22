import contextlib
import email
from unittest import mock
import uuid
from heat.common import exception as exc
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_handle_create_with_group(self):
    self._test_create(group='script')