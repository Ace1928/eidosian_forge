from . import errors, registry
from .branch import Branch
from .repository import Repository
from .revision import Revision
def parse_revision_id(self, revid):
    """Parse a bzr revision and return the matching mapping and foreign
        revid.

        :param revid: The bzr revision id
        :return: tuple with foreign revid and vcs mapping
        """
    if b':' not in revid or b'-' not in revid:
        raise errors.InvalidRevisionId(revid, None)
    try:
        foreign_vcs = self.get(revid.split(b'-')[0].decode('ascii'))
    except KeyError:
        raise errors.InvalidRevisionId(revid, None)
    return foreign_vcs.mapping_registry.revision_id_bzr_to_foreign(revid)