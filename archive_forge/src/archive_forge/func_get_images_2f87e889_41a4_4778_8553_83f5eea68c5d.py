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
def get_images_2f87e889_41a4_4778_8553_83f5eea68c5d(self, **kw):
    return (200, {}, {'image': self.get_images()[2]['images'][4]})