import re
from requests import cookies, utils
from . import _digest_auth_compat as auth
def init_per_thread_state(self):
    try:
        super(HTTPProxyDigestAuth, self).init_per_thread_state()
    except AttributeError:
        pass