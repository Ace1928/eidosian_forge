from breezy import errors, transport
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
Passing a child id will raise NoSuchId.

        This is because the parent directory will have already been removed.
        