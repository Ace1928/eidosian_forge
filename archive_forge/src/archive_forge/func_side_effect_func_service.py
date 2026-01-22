import testtools
from unittest import mock
from troveclient.compat import auth
from troveclient.compat import exceptions
def side_effect_func_service(key):
    if key == 'type':
        return 'trove'
    elif key == 'name':
        return 'test_service_name'
    return None