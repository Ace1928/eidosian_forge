import sys
import os
import inspect
from . import copydir
from . import command
from paste.util.template import paste_script_template_renderer
def print_vars(cls, vars, indent=0):
    max_name = max([len(v.name) for v in vars])
    for var in vars:
        if var.description:
            print('%s%s%s  %s' % (' ' * indent, var.name, ' ' * (max_name - len(var.name)), var.description))
        else:
            print('  %s' % var.name)
        if var.default is not command.NoDefault:
            print('      default: %r' % var.default)
        if var.should_echo is True:
            print('      should_echo: %s' % var.should_echo)
    print()