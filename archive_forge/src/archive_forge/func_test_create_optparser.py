import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
def test_create_optparser(self):
    option_names = ['verbose', 'debug', 'auth_url', 'username', 'apikey', 'tenant_id', 'auth_type', 'service_type', 'service_name', 'service_type', 'service_name', 'service_url', 'region', 'insecure', 'token', 'secure', 'json', 'terse', 'hide-debug']
    oparser = common.CliOptions.create_optparser(True)
    for option_name in option_names:
        self.check_option(oparser, option_name)
    oparser = common.CliOptions.create_optparser(False)
    for option_name in option_names:
        self.check_option(oparser, option_name)