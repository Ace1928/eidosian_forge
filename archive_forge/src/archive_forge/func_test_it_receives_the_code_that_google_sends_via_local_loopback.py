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
def test_it_receives_the_code_that_google_sends_via_local_loopback(self):
    expected_code = '1234ABC'
    received_code = self._do_first_sign_in(expected_code=expected_code, state=self.conn._state)
    self.assertEqual(received_code, expected_code)