import contextlib
import email
from unittest import mock
import uuid
from heat.common import exception as exc
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_get_message_multi_part(self):
    multipart = 'Content-Type: multipart/mixed; boundary="===============2579792489038011818=="\nMIME-Version: 1.0\n\n--===============2579792489038011818==\nContent-Type: text; charset="us-ascii"\nMIME-Version: 1.0\nContent-Transfer-Encoding: 7bit\nContent-Disposition: attachment;\n filename="/opt/stack/configure.d/55-heat-config"\n#!/bin/bash\n--===============2579792489038011818==--\n'
    parts = [{'config': '1e0e5a60-2843-4cfd-9137-d90bdf18eef5', 'type': 'multipart'}]
    self.init_config(parts=parts)
    self.rpc_client.show_software_config.return_value = {'config': multipart}
    result = self.config.get_message()
    self.assertEqual('1e0e5a60-2843-4cfd-9137-d90bdf18eef5', self.rpc_client.show_software_config.call_args[0][1])
    message = email.message_from_string(result)
    self.assertTrue(message.is_multipart())
    subs = message.get_payload()
    self.assertEqual(1, len(subs))
    self.assertEqual('#!/bin/bash', subs[0].get_payload())
    self.assertEqual('/opt/stack/configure.d/55-heat-config', subs[0].get_filename())