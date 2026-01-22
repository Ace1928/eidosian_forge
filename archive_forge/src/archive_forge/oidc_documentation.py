import abc
import base64
import hashlib
import os
import time
from urllib import parse as urlparse
import warnings
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import federation
Poll token endpoint for an access token.

        :param session: a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :param payload: a dict containing various OpenID Connect values,
                for example::
                {'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
                 'device_code': self.device_code}
        :type payload: dict
        