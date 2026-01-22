from dulwich.object_store import tree_lookup_path
from .. import osutils
from ..bzr.versionedfile import UnavailableRepresentation
from ..errors import NoSuchRevision
from ..graph import Graph
from ..revision import NULL_REVISION
from .mapping import decode_git_path, encode_git_path
def get_bytes_as(self, storage_kind):
    raise ValueError