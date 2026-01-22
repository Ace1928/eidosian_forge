import testtools
from unittest import mock
from troveclient.compat import auth
from troveclient.compat import exceptions
def side_effect_func(auth_url, body, root_key):
    return (auth_url, body, root_key)