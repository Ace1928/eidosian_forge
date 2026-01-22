import collections
from .objects import ZERO_SHA, format_timezone, parse_timezone
def drop_reflog_entry(f, index, rewrite=False):
    """Drop the specified reflog entry.

    Args:
        f: File-like object
        index: Reflog entry index (in Git reflog reverse 0-indexed order)
        rewrite: If a reflog entry's predecessor is removed, set its
            old SHA to the new SHA of the entry that now precedes it
    """
    if index < 0:
        raise ValueError('Invalid reflog index %d' % index)
    log = []
    offset = f.tell()
    for line in f:
        log.append((offset, parse_reflog_line(line)))
        offset = f.tell()
    inverse_index = len(log) - index - 1
    write_offset = log[inverse_index][0]
    f.seek(write_offset)
    if index == 0:
        f.truncate()
        return
    del log[inverse_index]
    if rewrite and index > 0 and log:
        if inverse_index == 0:
            previous_new = ZERO_SHA
        else:
            previous_new = log[inverse_index - 1][1].new_sha
        offset, entry = log[inverse_index]
        log[inverse_index] = (offset, Entry(previous_new, entry.new_sha, entry.committer, entry.timestamp, entry.timezone, entry.message))
    for _, entry in log[inverse_index:]:
        f.write(format_reflog_line(entry.old_sha, entry.new_sha, entry.committer, entry.timestamp, entry.timezone, entry.message))
    f.truncate()