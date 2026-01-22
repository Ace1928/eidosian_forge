import logging
import re
import warnings
from passlib import hash
from passlib.utils.compat import unicode
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE, PASS_TABLE_UTF8
def test_keyid_parameter(self):
    self.assertRaises(NotImplementedError, self.handler.verify, 'password', '$argon2i$v=19$m=65536,t=2,p=4,keyid=ABCD$c29tZXNhbHQ$IMit9qkFULCMA/ViizL57cnTLOa5DiVM9eMwpAvPwr4')