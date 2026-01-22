from __future__ import print_function, unicode_literals
from collections import defaultdict
import six
from pybtex.bibtex.builtins import builtins, print_warning
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.bibtex.utils import wrap
from pybtex.utils import CaseInsensitiveDict
def command_entry(self, fields, ints, strings):
    for id in fields:
        name = id.value()
        self.add_variable(name, Field(self, name))
    self.add_variable('crossref', Crossref(self))
    for id in ints:
        name = id.value()
        self.add_variable(name, EntryInteger(self, name))
    for id in strings:
        name = id.value()
        self.add_variable(name, EntryString(self, name))