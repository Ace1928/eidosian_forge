from unittest import mock
import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import users
def test_update_password_with_bad_inputs(self):
    old_password = uuid.uuid4().hex
    new_password = uuid.uuid4().hex
    self.assertRaises(exceptions.ValidationError, self.manager.update_password, old_password, None)
    self.assertRaises(exceptions.ValidationError, self.manager.update_password, old_password, '')
    self.assertRaises(exceptions.ValidationError, self.manager.update_password, None, new_password)
    self.assertRaises(exceptions.ValidationError, self.manager.update_password, '', new_password)
    self.assertRaises(exceptions.ValidationError, self.manager.update_password, None, None)
    self.assertRaises(exceptions.ValidationError, self.manager.update_password, '', '')
    password = uuid.uuid4().hex
    self.assertRaises(exceptions.ValidationError, self.manager.update_password, password, password)