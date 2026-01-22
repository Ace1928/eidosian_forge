from unittest import mock
from novaclient import client as base_client
from novaclient import exceptions as nova_exceptions
import requests
from urllib import parse as urlparse
from heat.tests import fakes
def get_images_1(self, **kw):
    return (200, {'image': self.get_images_detail()[1]['images'][0]})