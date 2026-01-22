def removeLeadingTrailingBlanks(s):
    lines = removeLeadingBlanks(s.split('\n'))
    lines.reverse()
    lines = removeLeadingBlanks(lines)
    lines.reverse()
    return '\n'.join(lines) + '\n'