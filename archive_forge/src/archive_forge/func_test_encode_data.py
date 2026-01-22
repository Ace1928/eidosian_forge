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
def test_encode_data(self):
    data = {'key': 'value'}
    json_data = '{"key": "value"}'
    encoded_data = self.conn.encode_data(data)
    self.assertEqual(encoded_data, json_data)