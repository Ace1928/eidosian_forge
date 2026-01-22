import os
import sys
from breezy import branch, osutils, registry, tests
def test_get_prefix(self):
    my_registry = registry.Registry()
    http_object = object()
    sftp_object = object()
    my_registry.register('http:', http_object)
    my_registry.register('sftp:', sftp_object)
    found_object, suffix = my_registry.get_prefix('http://foo/bar')
    self.assertEqual('//foo/bar', suffix)
    self.assertIs(http_object, found_object)
    self.assertIsNot(sftp_object, found_object)
    found_object, suffix = my_registry.get_prefix('sftp://baz/qux')
    self.assertEqual('//baz/qux', suffix)
    self.assertIs(sftp_object, found_object)