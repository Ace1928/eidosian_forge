import os
import sys
import logging
import tempfile
from unittest.mock import patch
import libcloud
from libcloud import _init_once
from libcloud.base import DriverTypeNotFoundError
from libcloud.test import unittest
from libcloud.utils.loggingconnection import LoggingConnection
def test_raises_error(self):
    with self.assertRaises(DriverTypeNotFoundError):
        libcloud.get_driver('potato', 'potato')