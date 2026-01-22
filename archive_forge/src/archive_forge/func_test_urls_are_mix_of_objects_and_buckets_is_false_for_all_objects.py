from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from gslib.exception import CommandException
from gslib import storage_url
from gslib.exception import InvalidUrlError
from gslib.tests.testcase import base
from unittest import mock
def test_urls_are_mix_of_objects_and_buckets_is_false_for_all_objects(self):
    urls = list(map(storage_url.StorageUrlFromString, ['gs://b/o', 'gs://b/p']))
    self.assertFalse(storage_url.UrlsAreMixOfBucketsAndObjects(urls))