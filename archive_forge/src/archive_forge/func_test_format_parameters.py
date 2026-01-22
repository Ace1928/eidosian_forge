import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_format_parameters(self):
    p = utils.format_parameters(['InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17'])
    self.assertEqual({'InstanceType': 'm1.large', 'DBUsername': 'wp', 'DBPassword': 'verybadpassword', 'KeyName': 'heat_key', 'LinuxDistribution': 'F17'}, p)