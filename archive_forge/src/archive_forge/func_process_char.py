import re
def process_char(char):
    if char == '\n':
        return '\\n'
    if char == '\\':
        return '\\\\'
    if char == '"':
        return '\\"'
    return char