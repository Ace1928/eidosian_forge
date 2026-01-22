import io
import os
import pathlib
import re
import sys
from pprint import pformat
from IPython.core import magic_arguments
from IPython.core import oinspect
from IPython.core import page
from IPython.core.alias import AliasError, Alias
from IPython.core.error import UsageError
from IPython.core.magic import  (
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.openpy import source_to_unicode
from IPython.utils.process import abbrev_cwd
from IPython.utils.terminal import set_term_title
from traitlets import Bool
from warnings import warn
@line_magic
def bookmark(self, parameter_s=''):
    """Manage IPython's bookmark system.

        %bookmark <name>       - set bookmark to current dir
        %bookmark <name> <dir> - set bookmark to <dir>
        %bookmark -l           - list all bookmarks
        %bookmark -d <name>    - remove bookmark
        %bookmark -r           - remove all bookmarks

        You can later on access a bookmarked folder with::

          %cd -b <name>

        or simply '%cd <name>' if there is no directory called <name> AND
        there is such a bookmark defined.

        Your bookmarks persist through IPython sessions, but they are
        associated with each profile."""
    opts, args = self.parse_options(parameter_s, 'drl', mode='list')
    if len(args) > 2:
        raise UsageError('%bookmark: too many arguments')
    bkms = self.shell.db.get('bookmarks', {})
    if 'd' in opts:
        try:
            todel = args[0]
        except IndexError as e:
            raise UsageError('%bookmark -d: must provide a bookmark to delete') from e
        else:
            try:
                del bkms[todel]
            except KeyError as e:
                raise UsageError("%%bookmark -d: Can't delete bookmark '%s'" % todel) from e
    elif 'r' in opts:
        bkms = {}
    elif 'l' in opts:
        bks = sorted(bkms)
        if bks:
            size = max(map(len, bks))
        else:
            size = 0
        fmt = '%-' + str(size) + 's -> %s'
        print('Current bookmarks:')
        for bk in bks:
            print(fmt % (bk, bkms[bk]))
    elif not args:
        raise UsageError('%bookmark: You must specify the bookmark name')
    elif len(args) == 1:
        bkms[args[0]] = os.getcwd()
    elif len(args) == 2:
        bkms[args[0]] = args[1]
    self.shell.db['bookmarks'] = bkms