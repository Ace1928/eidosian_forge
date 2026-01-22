import sys
import breezy
import breezy.errors as errors
import breezy.gpg
from breezy.bzr.inventory import Inventory
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.per_interrepository import TestCaseWithInterRepository
from breezy.workingtree import WorkingTree
def check_repo_format_for_funky_id_on_win32(repo):
    if not repo._format.supports_funky_characters and sys.platform == 'win32':
        raise TestSkipped('funky chars not allowed on this platform in repository %s' % repo.__class__.__name__)