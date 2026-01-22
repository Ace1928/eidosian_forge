import os
import sys
from .. import config, tests, trace
from ..transport.http import opt_ssl_ca_certs, ssl
def test_default_exists(self):
    """Check that the default we provide exists for the tested platform."""
    stack = self.get_stack('')
    self.assertPathExists(stack.get('ssl.ca_certs'))