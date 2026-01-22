import contextlib
import email
from unittest import mock
import uuid
from heat.common import exception as exc
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_get_message_non_uuid(self):
    parts = [{'config': 'http://192.168.122.36:8000/v1/waitcondition/'}]
    self.init_config(parts=parts)
    result = self.config.get_message()
    message = email.message_from_string(result)
    self.assertTrue(message.is_multipart())
    subs = message.get_payload()
    self.assertEqual(1, len(subs))
    self.assertEqual('http://192.168.122.36:8000/v1/waitcondition/', subs[0].get_payload())