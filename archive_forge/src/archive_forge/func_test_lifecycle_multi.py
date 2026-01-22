from mock import patch, Mock
import unittest
import time
from boto.exception import S3ResponseError
from boto.s3.connection import S3Connection
from boto.s3.bucketlogging import BucketLogging
from boto.s3.lifecycle import Lifecycle
from boto.s3.lifecycle import Transition
from boto.s3.lifecycle import Expiration
from boto.s3.lifecycle import Rule
from boto.s3.acl import Grant
from boto.s3.tagging import Tags, TagSet
from boto.s3.website import RedirectLocation
from boto.compat import unquote_str
def test_lifecycle_multi(self):
    date = '2022-10-12T00:00:00.000Z'
    sc = 'GLACIER'
    lifecycle = Lifecycle()
    lifecycle.add_rule('1', '1/', 'Enabled', 1)
    lifecycle.add_rule('2', '2/', 'Enabled', Expiration(days=2))
    lifecycle.add_rule('3', '3/', 'Enabled', Expiration(date=date))
    lifecycle.add_rule('4', '4/', 'Enabled', None, Transition(days=4, storage_class=sc))
    lifecycle.add_rule('5', '5/', 'Enabled', None, Transition(date=date, storage_class=sc))
    self.bucket.configure_lifecycle(lifecycle)
    readlifecycle = self.bucket.get_lifecycle_config()
    for rule in readlifecycle:
        if rule.id == '1':
            self.assertEqual(rule.prefix, '1/')
            self.assertEqual(rule.expiration.days, 1)
        elif rule.id == '2':
            self.assertEqual(rule.prefix, '2/')
            self.assertEqual(rule.expiration.days, 2)
        elif rule.id == '3':
            self.assertEqual(rule.prefix, '3/')
            self.assertEqual(rule.expiration.date, date)
        elif rule.id == '4':
            self.assertEqual(rule.prefix, '4/')
            self.assertEqual(rule.transition.days, 4)
            self.assertEqual(rule.transition.storage_class, sc)
        elif rule.id == '5':
            self.assertEqual(rule.prefix, '5/')
            self.assertEqual(rule.transition.date, date)
            self.assertEqual(rule.transition.storage_class, sc)
        else:
            self.fail('unexpected id %s' % rule.id)