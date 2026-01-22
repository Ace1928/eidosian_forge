import contextlib
import email
from unittest import mock
import uuid
from heat.common import exception as exc
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_get_message_empty_list(self):
    parts = []
    self.init_config(parts=parts)
    result = self.config.get_message()
    message = email.message_from_string(result)
    self.assertTrue(message.is_multipart())