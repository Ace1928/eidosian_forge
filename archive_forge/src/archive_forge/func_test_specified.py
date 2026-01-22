import os
import sys
from .. import config, tests, trace
from ..transport.http import opt_ssl_ca_certs, ssl
def test_specified(self):
    self.build_tree(['cacerts.pem'])
    path = os.path.join(self.test_dir, 'cacerts.pem')
    stack = self.get_stack('ssl.ca_certs = %s\n' % path)
    self.assertEqual(path, stack.get('ssl.ca_certs'))