from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
We need to use a merge base that makes sense.

        A
        | \
        B  D
        | \|
        C  E

        Rebasing E on C should result in:

        A -> B -> C -> D' -> E'

        Ancestry:
        A:
        B: A
        C: A, B
        D: A
        E: A, B, D
        D': A, B, C
        E': A, B, C, D'

        