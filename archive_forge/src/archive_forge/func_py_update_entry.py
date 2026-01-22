import bisect
import codecs
import contextlib
import errno
import operator
import os
import stat
import sys
import time
import zlib
from stat import S_IEXEC
from .. import (cache_utf8, config, debug, errors, lock, osutils, trace,
from . import inventory, static_tuple
from .inventorytree import InventoryTreeChange
def py_update_entry(state, entry, abspath, stat_value, _stat_to_minikind=DirState._stat_to_minikind):
    """Update the entry based on what is actually on disk.

    This function only calculates the sha if it needs to - if the entry is
    uncachable, or clearly different to the first parent's entry, no sha
    is calculated, and None is returned.

    :param state: The dirstate this entry is in.
    :param entry: This is the dirblock entry for the file in question.
    :param abspath: The path on disk for this file.
    :param stat_value: The stat value done on the path.
    :return: None, or The sha1 hexdigest of the file (40 bytes) or link
        target of a symlink.
    """
    try:
        minikind = _stat_to_minikind[stat_value.st_mode & 61440]
    except KeyError:
        return None
    packed_stat = pack_stat(stat_value)
    saved_minikind, saved_link_or_sha1, saved_file_size, saved_executable, saved_packed_stat = entry[1][0]
    if not isinstance(saved_minikind, bytes):
        raise TypeError(saved_minikind)
    if minikind == b'd' and saved_minikind == b't':
        minikind = b't'
    if minikind == saved_minikind and packed_stat == saved_packed_stat:
        if minikind == b'd':
            return None
        if saved_file_size == stat_value.st_size:
            return saved_link_or_sha1
    link_or_sha1 = None
    worth_saving = True
    if minikind == b'f':
        executable = state._is_executable(stat_value.st_mode, saved_executable)
        if state._cutoff_time is None:
            state._sha_cutoff_time()
        if stat_value.st_mtime < state._cutoff_time and stat_value.st_ctime < state._cutoff_time and (len(entry[1]) > 1) and (entry[1][1][0] != b'a'):
            link_or_sha1 = state._sha1_file(abspath)
            entry[1][0] = (b'f', link_or_sha1, stat_value.st_size, executable, packed_stat)
        else:
            entry[1][0] = (b'f', b'', stat_value.st_size, executable, DirState.NULLSTAT)
            worth_saving = False
    elif minikind == b'd':
        link_or_sha1 = None
        entry[1][0] = (b'd', b'', 0, False, packed_stat)
        if saved_minikind != b'd':
            block_index, entry_index, dir_present, file_present = state._get_block_entry_index(entry[0][0], entry[0][1], 0)
            state._ensure_block(block_index, entry_index, osutils.pathjoin(entry[0][0], entry[0][1]))
        else:
            worth_saving = False
    elif minikind == b'l':
        if saved_minikind == b'l':
            worth_saving = False
        link_or_sha1 = state._read_link(abspath, saved_link_or_sha1)
        if state._cutoff_time is None:
            state._sha_cutoff_time()
        if stat_value.st_mtime < state._cutoff_time and stat_value.st_ctime < state._cutoff_time:
            entry[1][0] = (b'l', link_or_sha1, stat_value.st_size, False, packed_stat)
        else:
            entry[1][0] = (b'l', b'', stat_value.st_size, False, DirState.NULLSTAT)
    if worth_saving:
        state._mark_modified([entry])
    return link_or_sha1