import stat
from base64 import standard_b64decode
from dulwich.objects import Blob, Tree
def get_pristine_tar_tree(repo):
    """Retrieve the pristine tar tree for a repository.

    """
    try:
        cid = repo.refs[b'refs/heads/pristine-tar']
    except KeyError:
        return Tree()
    tid = repo.object_store[cid].tree
    return repo.object_store[tid]