import os
import pkg_resources
def parse_lines(data):
    result = []
    for line in data.splitlines():
        line = line.strip()
        if line and (not line.startswith('#')):
            result.append(line)
    return result