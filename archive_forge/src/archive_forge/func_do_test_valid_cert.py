import os
import ssl
import unittest
import mock
from nose.plugins.attrib import attr
import boto
from boto.pyami.config import Config
from boto import exception, https_connection
from boto.gs.connection import GSConnection
from boto.s3.connection import S3Connection
def do_test_valid_cert(self):
    self.assertConnectionThrows(S3Connection, exception.S3ResponseError)
    self.assertConnectionThrows(GSConnection, exception.GSResponseError)