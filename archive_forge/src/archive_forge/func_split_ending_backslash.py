import re
def split_ending_backslash(line):
    if len(line) > 0 and line[-1] == '\\':
        return (line[:-1], '\\\\\n')
    return (line, '')