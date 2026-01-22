import importlib.metadata
import logging
import warnings
from debtcollector import removals
from debtcollector import renames
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
import packaging.version
import requests
from keystoneclient import _discover
from keystoneclient import access
from keystoneclient.auth import base
from keystoneclient import baseclient
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient import session as client_session
def get_auth_ref_from_keyring(self, **kwargs):
    """Retrieve auth_ref from keyring.

        If auth_ref is found in keyring, (keyring_key, auth_ref) is returned.
        Otherwise, (keyring_key, None) is returned.

        :returns: (keyring_key, auth_ref) or (keyring_key, None)
        :returns: or (None, None) if use_keyring is not set in the object

        """
    keyring_key = None
    auth_ref = None
    if self.use_keyring:
        keyring_key = self._build_keyring_key(**kwargs)
        try:
            auth_ref = keyring.get_password('keystoneclient_auth', keyring_key)
            if auth_ref:
                auth_ref = pickle.loads(auth_ref)
                if auth_ref.will_expire_soon(self.stale_duration):
                    auth_ref = None
        except Exception as e:
            auth_ref = None
            _logger.warning('Unable to retrieve token from keyring %s', e)
    return (keyring_key, auth_ref)