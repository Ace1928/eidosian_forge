import json
from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
import requests
from heat.api.aws import ec2token
from heat.api.aws import exception
from heat.common import wsgi
from heat.tests import common
from heat.tests import utils
def test_filter_factory_none_app(self):
    ec2_filter = ec2token.EC2Token_filter_factory(global_conf={})
    self.assertIsNone(ec2_filter(None).application)