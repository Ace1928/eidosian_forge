import os
import unittest
import time
from boto.compat import StringIO
import mock
import boto
from boto.s3.connection import S3Connection
def test_etag_of_parts(self):
    key_name = 'etagtest'
    mpu = self.bucket.initiate_multipart_upload(key_name)
    fp = StringIO('small file')
    uparts = []
    uparts.append(mpu.upload_part_from_file(fp, part_num=1, size=5))
    uparts.append(mpu.upload_part_from_file(fp, part_num=2))
    fp.close()
    pn = 0
    for lpart in mpu:
        self.assertEqual(uparts[pn].etag, lpart.etag)
        pn += 1
    mpu.cancel_upload()