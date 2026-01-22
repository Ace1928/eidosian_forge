import abc
import base64
import functools
import hashlib
import json
import threading
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
def get_sp_url(self, session, sp_id, **kwargs):
    try:
        return self.get_access(session).service_providers.get_sp_url(sp_id)
    except exceptions.ServiceProviderNotFound:
        return None