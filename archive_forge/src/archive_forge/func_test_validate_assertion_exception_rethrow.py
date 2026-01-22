import collections
import copy
import datetime
import json
import logging
import time
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from heat.common import context
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.db import api as db_api
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.engine import update
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import stack as stack_object
from heat.objects import stack_tag as stack_tag_object
from heat.objects import user_creds as ucreds_object
from heat.tests import common
from heat.tests import fakes
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
@mock.patch.object(function, 'validate')
def test_validate_assertion_exception_rethrow(self, func_val):
    expected_msg = 'Expected Assertion Error'
    with mock.patch('heat.engine.stack.dependencies', new_callable=mock.PropertyMock) as mock_dependencies:
        mock_dependency = mock.MagicMock()
        mock_dependency.name = 'res'
        mock_dependency.external_id = None
        mock_dependency.validate.side_effect = AssertionError(expected_msg)
        mock_dependencies.Dependencies.return_value = [mock_dependency]
        stc = stack.Stack(self.ctx, utils.random_name(), self.tmpl)
        mock_res = mock.Mock()
        mock_res.name = mock_dependency.name
        mock_res.t = mock.Mock()
        mock_res.t.name = mock_res.name
        stc._resources = {mock_res.name: mock_res}
        expected_exception = self.assertRaises(AssertionError, stc.validate)
        self.assertEqual(expected_msg, str(expected_exception))
        mock_dependency.validate.assert_called_once_with()
    tmpl = template_format.parse("\n        HeatTemplateFormatVersion: '2012-12-12'\n        Outputs:\n          foo:\n            Value: bar\n        ")
    stc = stack.Stack(self.ctx, utils.random_name(), template.Template(tmpl))
    func_val.side_effect = AssertionError(expected_msg)
    expected_exception = self.assertRaises(AssertionError, stc.validate)
    self.assertEqual(expected_msg, str(expected_exception))