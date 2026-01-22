from mock import patch, Mock
import unittest
from boto.s3.bucket import ResultSet
from boto.s3.bucketlistresultset import multipart_upload_lister
from boto.s3.bucketlistresultset import versioned_bucket_lister
def test_list_object_versions_with_url_encoding(self):
    self._test_patched_lister_encoding('get_all_versions', versioned_bucket_lister)