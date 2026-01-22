import pickle
import re
from debian.deprecation import function_deprecated_by
def parse_tags(input_data):
    lre = re.compile('^(.+?)(?::?\\s*|:\\s+(.+?)\\s*)$')
    for line in input_data:
        m = lre.match(line)
        if not m:
            continue
        pkgs = set(m.group(1).split(', '))
        if m.group(2):
            tags = set(m.group(2).split(', '))
        else:
            tags = set()
        yield (pkgs, tags)