import logging
import re
import warnings
from passlib import hash
from passlib.utils.compat import unicode
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE, PASS_TABLE_UTF8
def random_memory_cost(self):
    if self.test.backend == 'argon2pure':
        return self.randintgauss(128, 384, 256, 128)
    else:
        return self.randintgauss(128, 32767, 16384, 4096)