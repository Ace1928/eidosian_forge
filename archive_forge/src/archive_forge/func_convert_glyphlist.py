def convert_glyphlist(path: str) -> None:
    """Convert a glyph list into a python representation.

    See output below.
    """
    state = 0
    with open(path, 'r') as fileinput:
        for line in fileinput.readlines():
            line = line.strip()
            if not line or line.startswith('#'):
                if state == 1:
                    state = 2
                    print('}\n')
                print(line)
                continue
            if state == 0:
                print('\nglyphname2unicode = {')
                state = 1
            name, x = line.split(';')
            codes = x.split(' ')
            print(" {!r}: u'{}',".format(name, ''.join(('\\u%s' % code for code in codes))))