from unittest import mock
from openstack.orchestration.v1 import template
from openstack import resource
from openstack.tests.unit import base
@mock.patch.object(resource.Resource, '_translate_response')
def test_validate_with_env(self, mock_translate):
    sess = mock.Mock()
    sot = template.Template()
    tmpl = mock.Mock()
    env = mock.Mock()
    body = {'template': tmpl, 'environment': env}
    sot.validate(sess, tmpl, environment=env)
    sess.post.assert_called_once_with('/validate', json=body)
    mock_translate.assert_called_once_with(sess.post.return_value)