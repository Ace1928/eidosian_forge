import testtools
from unittest import mock
from troveclient.compat import auth
from troveclient.compat import exceptions
def test_get_authenticator_cls(self):
    class_list = (auth.KeyStoneV2Authenticator, auth.Auth1_1, auth.FakeAuth)
    for c in class_list:
        self.assertEqual(c, auth.get_authenticator_cls(c))
    class_names = {'keystone': auth.KeyStoneV3Authenticator, 'auth1.1': auth.Auth1_1, 'fake': auth.FakeAuth}
    for cn in class_names.keys():
        self.assertEqual(class_names[cn], auth.get_authenticator_cls(cn))
    cls_or_name = '_unknown_'
    self.assertRaises(ValueError, auth.get_authenticator_cls, cls_or_name)