import copy
import datetime
import re
from unittest import mock
from urllib import parse
from oslo_utils import strutils
import novaclient
from novaclient import api_versions
from novaclient import client as base_client
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils
from novaclient.v2 import client
def get_images_55bb23af_97a4_4068_bdf8_f10c62880ddf(self, **kw):
    return (200, {}, {'image': self.get_images()[2]['images'][1]})