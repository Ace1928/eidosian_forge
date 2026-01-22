from unittest import mock
import uuid
from oslo_config import cfg
from troveclient import exceptions as troveexc
from troveclient.v1 import users
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import trove
from heat.engine import resource
from heat.engine.resources.openstack.trove import instance as dbinstance
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def test_check_complete_users(self, mock_client, mock_plugin):
    mock_instance = mock.Mock(status='ACTIVE', name='mock_instance')
    mock_client().instances.get.return_value = mock_instance
    mock_plugin().is_client_exception.return_value = False
    mock_client().users.get.return_value = users.User(None, {'databases': [{'name': 'bar'}, {'name': 'buzz'}], 'name': 'baz'}, loaded=True)
    updates = {'users': [{'databases': ['bar', 'biff'], 'name': 'baz', 'password': 'changed'}, {'ACTION': 'CREATE', 'databases': ['biff', 'bar'], 'name': 'user2', 'password': 'password'}, {'ACTION': 'DELETE', 'name': 'deleted'}]}
    trove = dbinstance.Instance('test', self._rdef, self._stack)
    self.assertTrue(trove.check_update_complete(updates))
    create_calls = [mock.call(mock_instance, [{'password': 'password', 'databases': [{'name': 'biff'}, {'name': 'bar'}], 'name': 'user2'}])]
    delete_calls = [mock.call(mock_instance, 'deleted')]
    mock_client().users.create.assert_has_calls(create_calls)
    mock_client().users.delete.assert_has_calls(delete_calls)
    self.assertEqual(1, mock_client().users.create.call_count)
    self.assertEqual(1, mock_client().users.delete.call_count)
    updateattr = mock_client().users.update_attributes
    updateattr.assert_called_once_with(mock_instance, 'baz', newuserattr={'password': 'changed'}, hostname=mock.ANY)
    mock_client().users.grant.assert_called_once_with(mock_instance, 'baz', ['biff'])
    mock_client().users.revoke.assert_called_once_with(mock_instance, 'baz', ['buzz'])