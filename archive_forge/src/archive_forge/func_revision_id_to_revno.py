import os
from breezy import tests
from breezy.bzr.tests.matchers import ContainsNoVfsCalls
from breezy.errors import NoSuchRevision
def revision_id_to_revno(s, r):
    raise NoSuchRevision(s, r)