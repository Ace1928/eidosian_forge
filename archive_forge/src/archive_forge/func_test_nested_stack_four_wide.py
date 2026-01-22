from unittest import mock
from oslo_config import cfg
from requests import exceptions
import yaml
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import api
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.cfn import stack as stack_res
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import resource_data as resource_data_object
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_nested_stack_four_wide(self):
    root_template = "\nHeatTemplateFormatVersion: 2012-12-12\nResources:\n    Nested:\n        Type: AWS::CloudFormation::Stack\n        Properties:\n            TemplateURL: 'https://server.test/depth1.template'\n            Parameters:\n                KeyName: foo\n    Nested2:\n        Type: AWS::CloudFormation::Stack\n        Properties:\n            TemplateURL: 'https://server.test/depth2.template'\n            Parameters:\n                KeyName: foo\n    Nested3:\n        Type: AWS::CloudFormation::Stack\n        Properties:\n            TemplateURL: 'https://server.test/depth3.template'\n            Parameters:\n                KeyName: foo\n    Nested4:\n        Type: AWS::CloudFormation::Stack\n        Properties:\n            TemplateURL: 'https://server.test/depth4.template'\n            Parameters:\n                KeyName: foo\n"
    urlfetch.get.return_value = self.nested_template
    self.validate_stack(root_template)
    calls = [mock.call('https://server.test/depth1.template'), mock.call('https://server.test/depth2.template'), mock.call('https://server.test/depth3.template'), mock.call('https://server.test/depth4.template')]
    urlfetch.get.assert_has_calls(calls, any_order=True)