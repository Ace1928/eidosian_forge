import stat
from base64 import standard_b64decode
from dulwich.objects import Blob, Tree
def read_git_pristine_tar_data(repo, filename):
    """Read pristine data from a Git repository.

    :param repo: Git repository to read from
    :param filename: Name of file to read
    :return: Tuple with delta and id
    """
    tree = get_pristine_tar_tree(repo)
    delta = tree[filename + b'.delta'][1]
    gitid = tree[filename + b'.id'][1]
    return (repo.object_store[delta].data, repo.object_store[gitid].data)