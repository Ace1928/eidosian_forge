import stat
from base64 import standard_b64decode
from dulwich.objects import Blob, Tree
Add pristine tar data to a Git repository.

    :param repo: Git repository to add data to
    :param filename: Name of file to store for
    :param delta: pristine-tar delta
    :param gitid: Git id the pristine tar delta is generated against
    