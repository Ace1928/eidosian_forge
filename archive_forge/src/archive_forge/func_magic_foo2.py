import argparse
import sys
from IPython.core.magic_arguments import (argument, argument_group, kwds,
@magic_arguments()
def magic_foo2(self, args):
    """ A docstring.
    """
    return parse_argstring(magic_foo2, args)