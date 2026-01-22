import logging
import re
import warnings
from passlib import hash
from passlib.utils.compat import unicode
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE, PASS_TABLE_UTF8
def test_needs_update_w_version(self):
    handler = self.handler.using(memory_cost=65536, time_cost=2, parallelism=4, digest_size=32)
    hash = '$argon2i$m=65536,t=2,p=4$c29tZXNhbHQAAAAAAAAAAA$QWLzI4TY9HkL2ZTLc8g6SinwdhZewYrzz9zxCo0bkGY'
    if handler.max_version == 16:
        self.assertFalse(handler.needs_update(hash))
    else:
        self.assertTrue(handler.needs_update(hash))