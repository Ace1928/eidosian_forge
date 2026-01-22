from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as mistral_client
from heat.engine.resources.openstack.vitrage.vitrage_template import \
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_vitrage_validate_wrong_format(self):
    """wrong result format for vitrage templete validate"""
    template = VitrageTemplate('execute_healing', self.resource_def, self.stack)
    self.vitrage.template.validate.return_value = {}
    self.assertRaises(exception.StackValidationFailed, scheduler.TaskRunner(template.validate))
    self.vitrage.template.validate.return_value = {'results': []}
    self.assertRaises(exception.StackValidationFailed, scheduler.TaskRunner(template.validate))
    self.vitrage.template.validate.return_value = {'results': [{'status': 'validation OK', 'file path': '/tmp/tmpNUEgE3', 'message': 'Template validation is OK', 'status code': 0, 'description': 'Template validation'}, {'status': 'validation OK', 'file path': '/tmp/tmpNUEgE3', 'message': 'Template validation is OK', 'status code': 0, 'description': 'Template validation'}]}
    self.assertRaises(exception.StackValidationFailed, scheduler.TaskRunner(template.validate))
    self.vitrage.template.validate.return_value = {'results': [{'status': 'validation OK', 'file path': '/tmp/tmpNUEgE3', 'message': 'Template validation is OK', 'description': 'Template validation'}]}
    self.assertRaises(exception.StackValidationFailed, scheduler.TaskRunner(template.validate))