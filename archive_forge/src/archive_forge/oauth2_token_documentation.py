import webob
from oslo_log import log as logging
from oslo_serialization import jsonutils
from keystonemiddleware.auth_token import _user_plugin
from keystonemiddleware.auth_token import AuthProtocol
from keystonemiddleware import exceptions
from keystonemiddleware.i18n import _
Process request.

        :param request: Incoming request
        :type request: _request.AuthTokenRequest
        