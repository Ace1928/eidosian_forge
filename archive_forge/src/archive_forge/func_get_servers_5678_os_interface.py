from unittest import mock
from novaclient import client as base_client
from novaclient import exceptions as nova_exceptions
import requests
from urllib import parse as urlparse
from heat.tests import fakes
def get_servers_5678_os_interface(self, **kw):
    return (200, {'interfaceAttachments': [{'fixed_ips': [{'ip_address': '10.0.0.1', 'subnet_id': 'f8a6e8f8-c2ec-497c-9f23-da9616de54ef'}], 'port_id': 'ce531f90-199f-48c0-816c-13e38010b442'}]})