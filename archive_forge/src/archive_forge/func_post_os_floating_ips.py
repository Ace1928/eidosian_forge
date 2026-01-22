from unittest import mock
from novaclient import client as base_client
from novaclient import exceptions as nova_exceptions
import requests
from urllib import parse as urlparse
from heat.tests import fakes
def post_os_floating_ips(self, body, **kw):
    return (202, self.get_os_floating_ips_1()[1])