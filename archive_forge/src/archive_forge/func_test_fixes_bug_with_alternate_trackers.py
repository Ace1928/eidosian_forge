import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_fixes_bug_with_alternate_trackers(self):
    """--fixes can be used on a properly configured branch to mark bug
        fixes on multiple trackers.
        """
    tree = self.make_branch_and_tree('tree')
    tree.branch.get_config().set_user_option('trac_twisted_url', 'http://twistedmatrix.com/trac')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    self.run_bzr('commit -m hello --fixes=lp:123 --fixes=twisted:235 tree/')
    last_rev = tree.branch.repository.get_revision(tree.last_revision())
    properties = dict(last_rev.properties)
    del properties['branch-nick']
    self.assertEqual({'bugs': 'https://launchpad.net/bugs/123 fixed\nhttp://twistedmatrix.com/trac/ticket/235 fixed'}, properties)