import errno
import os
from io import BytesIO
from typing import Set
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import bedding
def tree_ignores_add_patterns(tree, name_pattern_list):
    """Add more ignore patterns to the ignore file in a tree.
    If ignore file does not exist then it will be created.
    The ignore file will be automatically added under version control.

    :param tree: Working tree to update the ignore list.
    :param name_pattern_list: List of ignore patterns.
    :return: None
    """
    ifn = tree.abspath(tree._format.ignore_filename)
    if tree.has_filename(ifn):
        with open(ifn, 'rb') as f:
            file_contents = f.read()
            if file_contents.find(b'\r\n') != -1:
                newline = b'\r\n'
            else:
                newline = b'\n'
    else:
        file_contents = b''
        newline = os.linesep.encode()
    with BytesIO(file_contents) as sio:
        ignores = parse_ignore_file(sio)
    with atomicfile.AtomicFile(ifn, 'wb') as f:
        f.write(file_contents)
        if len(file_contents) > 0 and (not file_contents.endswith(b'\n')):
            f.write(newline)
        for pattern in name_pattern_list:
            if pattern not in ignores:
                f.write(pattern.encode('utf-8'))
                f.write(newline)
    if not tree.is_versioned(tree._format.ignore_filename):
        tree.add([tree._format.ignore_filename])