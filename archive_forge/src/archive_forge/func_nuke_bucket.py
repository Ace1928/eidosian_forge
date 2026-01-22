from tests.unit import unittest
import time
import boto
from boto.s3.connection import S3Connection, Location
def nuke_bucket(self, bucket):
    for key in bucket:
        key.delete()
    bucket.delete()