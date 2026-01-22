def rest(name):
    """Convert name to reStructuredText."""
    s = ''
    while name:
        c = name[0]
        if c == '_':
            s += '\\ :sub:`%s`\\ ' % name[1]
            name = name[2:]
        elif c == '^':
            s += '\\ :sup:`%s`\\ ' % name[1]
            name = name[2:]
        else:
            s += c
            name = name[1:]
    return s