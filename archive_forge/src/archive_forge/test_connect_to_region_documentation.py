from tests.compat import unittest
from boto.s3.connection import S3Connection
from boto.s3 import connect_to_region

Unit test for passing in 'host' parameter and overriding the region
See issue: #2522
