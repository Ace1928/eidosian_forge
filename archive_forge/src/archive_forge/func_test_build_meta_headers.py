from unittest import mock
import swiftclient.client as sc
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine.resources.openstack.swift import container as swift_c
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_build_meta_headers(self):
    headers = {'Web-Index': 'index.html', 'Web-Error': 'error.html'}
    self.assertEqual({}, swift_c.SwiftContainer._build_meta_headers('container', {}))
    self.assertEqual({}, swift_c.SwiftContainer._build_meta_headers('container', None))
    built = swift_c.SwiftContainer._build_meta_headers('container', headers)
    expected = {'X-Container-Meta-Web-Index': 'index.html', 'X-Container-Meta-Web-Error': 'error.html'}
    self.assertEqual(expected, built)