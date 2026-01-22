from stat import S_ISDIR
from git.compat import safe_decode, defenc
from typing import (
def traverse_tree_recursive(odb: 'GitCmdObjectDB', tree_sha: bytes, path_prefix: str) -> List[EntryTup]:
    """
    :return: list of entries of the tree pointed to by the binary tree_sha.

        An entry has the following format:

        * [0] 20 byte sha
        * [1] mode as int
        * [2] path relative to the repository

    :param path_prefix: Prefix to prepend to the front of all returned paths.
    """
    entries = []
    data = tree_entries_from_data(odb.stream(tree_sha).read())
    for sha, mode, name in data:
        if S_ISDIR(mode):
            entries.extend(traverse_tree_recursive(odb, sha, path_prefix + name + '/'))
        else:
            entries.append((sha, mode, path_prefix + name))
    return entries