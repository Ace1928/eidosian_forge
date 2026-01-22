import abc
import collections.abc
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import sexp
from . import conversion
import rpy2.rlike.container as rlc
import datetime
import copy
import itertools
import math
import os
import jinja2  # type: ignore
import time
import tzlocal
from time import struct_time, mktime
import typing
import warnings
from rpy2.rinterface import (Sexp, ListSexpVector, StrSexpVector,
def to_csvfile(self, path, quote=True, sep=',', eol=os.linesep, na='NA', dec='.', row_names=True, col_names=True, qmethod='escape', append=False):
    """ Save the data into a .csv file.

        :param path         : string with a path
        :param quote        : quote character
        :param sep          : separator character
        :param eol          : end-of-line character(s)
        :param na           : string for missing values
        :param dec          : string for decimal separator
        :param row_names    : boolean (save row names, or not)
        :param col_names    : boolean (save column names, or not)
        :param comment_char : method to 'escape' special characters
        :param append       : boolean (append if the file in the path is
        already existing, or not)
        """
    cv = conversion.get_conversion()
    path = cv.py2rpy(path)
    append = cv.py2rpy(append)
    sep = cv.py2rpy(sep)
    eol = cv.py2rpy(eol)
    na = cv.py2rpy(na)
    dec = cv.py2rpy(dec)
    row_names = cv.py2rpy(row_names)
    col_names = cv.py2rpy(col_names)
    qmethod = cv.py2rpy(qmethod)
    res = self._write_table(self, **{'file': path, 'quote': quote, 'sep': sep, 'eol': eol, 'na': na, 'dec': dec, 'row.names': row_names, 'col.names': col_names, 'qmethod': qmethod, 'append': append})
    return res