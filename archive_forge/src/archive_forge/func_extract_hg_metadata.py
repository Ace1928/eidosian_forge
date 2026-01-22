import urllib.parse
def extract_hg_metadata(message):
    """Extract Mercurial metadata from a commit message.

    :param message: Commit message to extract from
    :return: Tuple with original commit message, renames, branch and
        extra data.
    """
    split = message.split('\n--HG--\n', 1)
    renames = {}
    extra = {}
    branch = None
    if len(split) == 2:
        message, meta = split
        lines = meta.split('\n')
        for line in lines:
            if line == '':
                continue
            command, data = line.split(' : ', 1)
            if command == 'rename':
                before, after = data.split(' => ', 1)
                renames[after] = before
            elif command == 'branch':
                branch = data
            elif command == 'extra':
                before, after = data.split(' : ', 1)
                extra[before] = urllib.parse.unquote(after)
            else:
                raise KeyError('unknown hg-git metadata command %s' % command)
    return (message, renames, branch, extra)