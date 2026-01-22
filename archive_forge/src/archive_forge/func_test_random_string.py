import re
from unittest import mock
from testtools import matchers
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_random_string(self):
    stack = self.create_stack(self.template_random_string)
    secret1 = stack['secret1']
    random_string = secret1.FnGetAtt('value')
    self.assert_min('[a-zA-Z0-9]', random_string, 32)
    self.assertRaises(exception.InvalidTemplateAttribute, secret1.FnGetAtt, 'foo')
    self.assertEqual(secret1.FnGetRefId(), random_string)
    secret2 = stack['secret2']
    random_string = secret2.FnGetAtt('value')
    self.assert_min('[a-zA-Z0-9]', random_string, 10)
    self.assertEqual(secret2.FnGetRefId(), random_string)
    secret3 = stack['secret3']
    random_string = secret3.FnGetAtt('value')
    self.assertEqual(32, len(random_string))
    self.assert_min('[0-9]', random_string, 1)
    self.assert_min('[A-Z]', random_string, 1)
    self.assert_min('[a-z]', random_string, 20)
    self.assert_min('[(),\\[\\]{}]', random_string, 1)
    self.assert_min('[$_]', random_string, 2)
    self.assert_min('@', random_string, 5)
    self.assertEqual(secret3.FnGetRefId(), random_string)
    secret4 = stack['secret4']
    random_string = secret4.FnGetAtt('value')
    self.assertEqual(25, len(random_string))
    self.assert_min('[0-9]', random_string, 1)
    self.assert_min('[A-Z]', random_string, 1)
    self.assert_min('[a-z]', random_string, 20)
    self.assertEqual(secret4.FnGetRefId(), random_string)
    secret5 = stack['secret5']
    random_string = secret5.FnGetAtt('value')
    self.assertEqual(10, len(random_string))
    self.assert_min('[(),\\[\\]{}]', random_string, 1)
    self.assert_min('[$_]', random_string, 2)
    self.assert_min('@', random_string, 5)
    self.assertEqual(secret5.FnGetRefId(), random_string)
    secret5.resource_id = None
    self.assertEqual('secret5', secret5.FnGetRefId())