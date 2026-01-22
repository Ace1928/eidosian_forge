import testtools
from unittest import mock
from troveclient.compat import auth
from troveclient.compat import exceptions
def check_url_none(test_case, auth_class):
    authObj = auth_class(url=None, type=auth_class, client=None, username=None, password=None, tenant=None)
    try:
        authObj.authenticate()
        test_case.fail('AuthUrlNotGiven exception expected')
    except exceptions.AuthUrlNotGiven:
        pass