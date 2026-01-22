import os
from collections import namedtuple
import re
import sqlite3
import typing
import warnings
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects.packages_utils import (get_packagepath,
from collections import OrderedDict
def pages(topic):
    """ Get help pages corresponding to a given topic. """
    res = list()
    for path in _libpaths():
        for name in _packages(**{'all.available': True, 'lib.loc': StrSexpVector((path,))}):
            pack = Package(name)
            try:
                page = pack.fetch(topic)
                res.append(page)
            except HelpNotFoundError:
                pass
    return tuple(res)