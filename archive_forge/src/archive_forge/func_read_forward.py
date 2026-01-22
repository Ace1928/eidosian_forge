import sys
def read_forward(handle):
    """Read through whitespaces, return the first non-whitespace line."""
    while True:
        line = handle.readline()
        if not line or (line and line.strip()):
            return line