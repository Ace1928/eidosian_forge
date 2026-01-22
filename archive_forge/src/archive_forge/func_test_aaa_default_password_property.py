import unittest
import logging
import time
def test_aaa_default_password_property(self):
    cls = self.test_model()
    obj = cls(id='passwordtest')
    obj.password = 'foo'
    self.assertEquals('foo', obj.password)
    obj.save()
    time.sleep(5)
    obj = cls.get_by_id('passwordtest')
    self.assertEquals('foo', obj.password)