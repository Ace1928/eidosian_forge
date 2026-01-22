from stat import S_ISDIR
from ... import controldir, errors, gpg, osutils, repository
from ... import revision as _mod_revision
from ... import tests, transport, ui
from ...tests import TestCaseWithTransport, TestNotApplicable, test_server
from ...transport import memory
from .. import inventory
from ..btree_index import BTreeGraphIndex
from ..groupcompress_repo import RepositoryFormat2a
from ..index import GraphIndex
from ..smart import client
def test_stack_checks_rich_root_compatibility(self):
    repo = self.make_repository('repo', format=self.get_format())
    if repo.supports_rich_root():
        if getattr(repo._format, 'supports_tree_reference', False):
            matching_format_name = '2a'
        elif repo._format.supports_chks:
            matching_format_name = '2a'
        else:
            matching_format_name = 'rich-root-pack'
        mismatching_format_name = 'pack-0.92'
    else:
        if repo._format.supports_chks:
            raise AssertionError('no non-rich-root CHK formats known')
        else:
            matching_format_name = 'pack-0.92'
        mismatching_format_name = 'pack-0.92-subtree'
    base = self.make_repository('base', format=matching_format_name)
    repo.add_fallback_repository(base)
    bad_repo = self.make_repository('mismatch', format=mismatching_format_name)
    e = self.assertRaises(errors.IncompatibleRepositories, repo.add_fallback_repository, bad_repo)
    self.assertContainsRe(str(e), '(?m)KnitPackRepository.*/mismatch/.*\\nis not compatible with\\n.*Repository.*/repo/.*\\ndifferent rich-root support')