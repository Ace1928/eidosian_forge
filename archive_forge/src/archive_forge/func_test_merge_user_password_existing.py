import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_merge_user_password_existing(self):
    url = 'http://josh:harlow@www.yahoo.com/'
    parsed = misc.parse_uri(url)
    existing = {'username': 'joe', 'password': 'biggie'}
    joined = misc.merge_uri(parsed, existing)
    self.assertEqual('www.yahoo.com', joined.get('hostname'))
    self.assertEqual('joe', joined.get('username'))
    self.assertEqual('biggie', joined.get('password'))