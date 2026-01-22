import os
from breezy.tests import TestCaseWithTransport
from breezy.version_info_formats import VersionInfoBuilder
def test_custom_no_clean_in_template(self):

    def should_not_be_called(self):
        raise AssertionError('Method on {!r} should not have been used'.format(self))
    self.overrideAttr(VersionInfoBuilder, '_extract_file_revisions', should_not_be_called)
    self.create_tree()
    out, err = self.run_bzr('version-info --custom --template=r{revno} branch')
    self.assertEqual('r2', out)
    self.assertEqual('', err)