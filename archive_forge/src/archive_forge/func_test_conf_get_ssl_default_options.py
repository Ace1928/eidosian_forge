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
def test_conf_get_ssl_default_options(self):
    ec2 = ec2token.EC2Token(app=None, conf={})
    self.assertTrue(ec2.ssl_options['verify'], 'SSL verify should be True by default')
    self.assertIsNone(ec2.ssl_options['cert'], 'SSL client cert should be None by default')