from unittest import mock
from novaclient import client as base_client
from novaclient import exceptions as nova_exceptions
import requests
from urllib import parse as urlparse
from heat.tests import fakes
def get_os_availability_zone(self, *kw):
    return (200, {'availabilityZoneInfo': [{'zoneName': 'nova1'}]})