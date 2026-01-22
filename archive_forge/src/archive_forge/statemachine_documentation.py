import keyword
import sys
import os
import types
import importlib
import pyparsing as pp
An importer designed using the mechanism defined in :pep:`302`. I read
    the PEP, and also used Doug Hellmann's PyMOTW article `Modules and
    Imports`_, as a pattern.

    .. _`Modules and Imports`: http://www.doughellmann.com/PyMOTW/sys/imports.html

    Define a subclass that specifies a :attr:`suffix` attribute, and
    implements a :meth:`process_filedata` method. Then call the classmethod
    :meth:`register` on your class to actually install it in the appropriate
    places in :mod:`sys`. 