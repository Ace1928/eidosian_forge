import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def do_alias(self, arg):
    """alias [name [command [parameter parameter ...] ]]
        Create an alias called 'name' that executes 'command'.  The
        command must *not* be enclosed in quotes.  Replaceable
        parameters can be indicated by %1, %2, and so on, while %* is
        replaced by all the parameters.  If no command is given, the
        current alias for name is shown. If no name is given, all
        aliases are listed.

        Aliases may be nested and can contain anything that can be
        legally typed at the pdb prompt.  Note!  You *can* override
        internal pdb commands with aliases!  Those internal commands
        are then hidden until the alias is removed.  Aliasing is
        recursively applied to the first word of the command line; all
        other words in the line are left alone.

        As an example, here are two useful aliases (especially when
        placed in the .pdbrc file):

        # Print instance variables (usage "pi classInst")
        alias pi for k in %1.__dict__.keys(): print("%1.",k,"=",%1.__dict__[k])
        # Print instance variables in self
        alias ps pi self
        """
    args = arg.split()
    if len(args) == 0:
        keys = sorted(self.aliases.keys())
        for alias in keys:
            self.message('%s = %s' % (alias, self.aliases[alias]))
        return
    if len(args) == 1:
        if args[0] in self.aliases:
            self.message('%s = %s' % (args[0], self.aliases[args[0]]))
        else:
            self.error(f"Unknown alias '{args[0]}'")
    else:
        self.aliases[args[0]] = ' '.join(args[1:])