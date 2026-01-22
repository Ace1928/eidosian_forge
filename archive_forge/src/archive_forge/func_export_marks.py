from ...trace import warning
def export_marks(filename, revision_ids):
    """Save marks to a file.

    :param filename: filename to save data to
    :param revision_ids: dictionary mapping marks -> bzr revision-ids
    """
    try:
        f = open(filename, 'wb')
    except OSError:
        warning('Could not open export-marks file %s - not exporting marks', filename)
        return
    try:
        for mark, revid in sorted(revision_ids.items()):
            f.write(b':%s %s\n' % (mark.lstrip(b':'), revid))
    finally:
        f.close()