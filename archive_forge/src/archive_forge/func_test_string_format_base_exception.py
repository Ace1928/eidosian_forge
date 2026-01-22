from unittest import mock
from oslotest import base
from aodhclient import exceptions
def test_string_format_base_exception(self):
    self.assertEqual('Unknown Error (HTTP N/A)', '%s' % exceptions.ClientException())