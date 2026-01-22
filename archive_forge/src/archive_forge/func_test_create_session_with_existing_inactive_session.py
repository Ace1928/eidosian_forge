from datetime import datetime
from unittest import mock
from eventlet import greenthread
from oslo_context import context
import suds
from oslo_vmware import api
from oslo_vmware import exceptions
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_create_session_with_existing_inactive_session(self):
    old_session_key = '12345'
    new_session_key = '67890'
    session = mock.Mock()
    session.key = new_session_key
    api_session = self._create_api_session(False)
    api_session._session_id = old_session_key
    api_session._session_username = api_session._server_username
    vim_obj = api_session.vim
    vim_obj.Login.return_value = session
    vim_obj.SessionIsActive.return_value = False
    api_session._create_session()
    session_manager = vim_obj.service_content.sessionManager
    vim_obj.SessionIsActive.assert_called_once_with(session_manager, sessionID=old_session_key, userName=VMwareAPISessionTest.USERNAME)
    vim_obj.Login.assert_called_once_with(session_manager, userName=VMwareAPISessionTest.USERNAME, password=VMwareAPISessionTest.PASSWORD, locale='en')
    self.assertEqual(new_session_key, api_session._session_id)