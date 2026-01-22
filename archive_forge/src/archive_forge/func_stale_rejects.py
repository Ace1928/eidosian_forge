import re
from requests import cookies, utils
from . import _digest_auth_compat as auth
@stale_rejects.setter
def stale_rejects(self, value):
    thread_local = getattr(self, '_thread_local', None)
    if thread_local is None:
        self._stale_rejects = value
    else:
        thread_local.stale_rejects = value