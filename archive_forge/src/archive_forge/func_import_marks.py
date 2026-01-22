from ...trace import warning
def import_marks(filename):
    """Read the mapping of marks to revision-ids from a file.

    :param filename: the file to read from
    :return: None if an error is encountered or a dictionary with marks
        as keys and revision-ids as values
    """
    try:
        f = open(filename, 'rb')
    except OSError:
        warning('Could not import marks file %s - not importing marks', filename)
        return None
    try:
        revision_ids = {}
        line = f.readline()
        if line == b'format=1\n':
            branch_names = {}
            for string in f.readline().rstrip(b'\n').split(b'\x00'):
                if not string:
                    continue
                name, integer = string.rsplit(b'.', 1)
                branch_names[name] = int(integer)
            line = f.readline()
        while line:
            line = line.rstrip(b'\n')
            mark, revid = line.split(b' ', 1)
            mark = mark.lstrip(b':')
            revision_ids[mark] = revid
            line = f.readline()
    finally:
        f.close()
    return revision_ids