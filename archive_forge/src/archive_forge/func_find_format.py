import itertools
from .. import errors, lockable_files, lockdir
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from ..repository import Repository, RepositoryFormat, format_registry
from . import bzrdir
@classmethod
def find_format(klass, a_bzrdir):
    """Return the format for the repository object in a_bzrdir.

        This is used by brz native formats that have a "format" file in
        the repository.  Other methods may be used by different types of
        control directory.
        """
    try:
        transport = a_bzrdir.get_repository_transport(None)
        format_string = transport.get_bytes('format')
    except _mod_transport.NoSuchFile:
        raise errors.NoRepositoryPresent(a_bzrdir)
    return klass._find_format(format_registry, 'repository', format_string)