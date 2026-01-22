import sys
import re
import os
from configparser import RawConfigParser
def parse_config(filename, dirs=None):
    if dirs:
        filenames = [os.path.join(d, filename) for d in dirs]
    else:
        filenames = [filename]
    config = RawConfigParser()
    n = config.read(filenames)
    if not len(n) >= 1:
        raise PkgNotFound('Could not find file(s) %s' % str(filenames))
    meta = parse_meta(config)
    vars = {}
    if config.has_section('variables'):
        for name, value in config.items('variables'):
            vars[name] = _escape_backslash(value)
    secs = [s for s in config.sections() if not s in ['meta', 'variables']]
    sections = {}
    requires = {}
    for s in secs:
        d = {}
        if config.has_option(s, 'requires'):
            requires[s] = config.get(s, 'requires')
        for name, value in config.items(s):
            d[name] = value
        sections[s] = d
    return (meta, vars, sections, requires)