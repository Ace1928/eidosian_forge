from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def fill_command(args=None):
    import sys
    import optparse
    import pkg_resources
    import os
    if args is None:
        args = sys.argv[1:]
    dist = pkg_resources.get_distribution('Paste')
    parser = optparse.OptionParser(version=coerce_text(dist), usage=_fill_command_usage)
    parser.add_option('-o', '--output', dest='output', metavar='FILENAME', help='File to write output to (default stdout)')
    parser.add_option('--env', dest='use_env', action='store_true', help='Put the environment in as top-level variables')
    options, args = parser.parse_args(args)
    if len(args) < 1:
        print('You must give a template filename')
        sys.exit(2)
    template_name = args[0]
    args = args[1:]
    vars = {}
    if options.use_env:
        vars.update(os.environ)
    for value in args:
        if '=' not in value:
            print('Bad argument: %r' % value)
            sys.exit(2)
        name, value = value.split('=', 1)
        if name.startswith('py:'):
            name = name[:3]
            value = eval(value)
        vars[name] = value
    if template_name == '-':
        template_content = sys.stdin.read()
        template_name = '<stdin>'
    else:
        with open(template_name, 'rb') as f:
            template_content = f.read()
    template = Template(template_content, name=template_name)
    result = template.substitute(vars)
    if options.output:
        with open(options.output, 'wb') as f:
            f.write(result)
    else:
        sys.stdout.write(result)