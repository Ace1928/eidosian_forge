import os
import sys
import time
import random
import urllib
import datetime
import unittest
import threading
from unittest import mock
import requests
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.google import (
def test_it_aborts_if_state_is_suspicious(self):
    received_code = self._do_first_sign_in(expected_code='1234ABC', state=self.conn._state + 'very suspicious')
    self.assertEqual(received_code, None)