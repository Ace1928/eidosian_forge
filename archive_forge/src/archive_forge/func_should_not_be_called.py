import os
from breezy.tests import TestCaseWithTransport
from breezy.version_info_formats import VersionInfoBuilder
def should_not_be_called(self):
    raise AssertionError('Method on {!r} should not have been used'.format(self))