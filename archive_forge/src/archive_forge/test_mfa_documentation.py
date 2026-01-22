import unittest
import time
from nose.plugins.attrib import attr
from boto.s3.connection import S3Connection
from boto.exception import S3ResponseError
from boto.s3.deletemarker import DeleteMarker

Some unit tests for S3 MfaDelete with versioning
