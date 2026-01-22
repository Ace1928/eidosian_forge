import contextlib
import email
from unittest import mock
import uuid
from heat.common import exception as exc
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_get_message_text_with_filename(self):
    parts = [{'config': '1e0e5a60-2843-4cfd-9137-d90bdf18eef5', 'type': 'text', 'filename': '/opt/stack/configure.d/55-heat-config'}]
    self.init_config(parts=parts)
    self.rpc_client.show_software_config.return_value = {'config': '#!/bin/bash'}
    result = self.config.get_message()
    self.assertEqual('1e0e5a60-2843-4cfd-9137-d90bdf18eef5', self.rpc_client.show_software_config.call_args[0][1])
    message = email.message_from_string(result)
    self.assertTrue(message.is_multipart())
    subs = message.get_payload()
    self.assertEqual(1, len(subs))
    self.assertEqual('#!/bin/bash', subs[0].get_payload())
    self.assertEqual(parts[0]['filename'], subs[0].get_filename())