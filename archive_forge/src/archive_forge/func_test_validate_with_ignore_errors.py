from unittest import mock
from openstack.orchestration.v1 import template
from openstack import resource
from openstack.tests.unit import base
@mock.patch.object(resource.Resource, '_translate_response')
def test_validate_with_ignore_errors(self, mock_translate):
    sess = mock.Mock()
    sot = template.Template()
    tmpl = mock.Mock()
    body = {'template': tmpl}
    sot.validate(sess, tmpl, ignore_errors='123,456')
    sess.post.assert_called_once_with('/validate?ignore_errors=123%2C456', json=body)
    mock_translate.assert_called_once_with(sess.post.return_value)