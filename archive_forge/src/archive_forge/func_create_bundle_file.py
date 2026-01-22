from io import BytesIO
import breezy.bzr.bzrdir
import breezy.mergeable
import breezy.transport
import breezy.urlutils
from ... import errors, tests
from ...tests.per_transport import transport_test_permutations
from ...tests.scenarios import load_tests_apply_scenarios
from ...tests.test_transport import TestTransportImplementation
from ..bundle.serializer import write_bundle
def create_bundle_file(test_case):
    test_case.build_tree(['tree/', 'tree/a', 'tree/subdir/'])
    format = breezy.bzr.bzrdir.BzrDirFormat.get_default_format()
    bzrdir = format.initialize('tree')
    repo = bzrdir.create_repository()
    branch = repo.controldir.create_branch()
    wt = branch.controldir.create_workingtree()
    wt.add(['a', 'subdir/'])
    wt.commit('new project', rev_id=b'commit-1')
    out = BytesIO()
    write_bundle(wt.branch.repository, wt.get_parent_ids()[0], b'null:', out)
    out.seek(0)
    return (out, wt)