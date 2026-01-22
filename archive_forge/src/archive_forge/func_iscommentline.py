def iscommentline(line):
    c = line.lstrip()[:1]
    return c in COMMENTCHARS