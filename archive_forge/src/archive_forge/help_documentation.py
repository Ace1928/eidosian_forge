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
 Fetch the documentation page associated with a given alias.

        For S4 classes, the class name is *often* suffixed with '-class'.
        For example, the alias to the documentation for the class
        AnnotatedDataFrame in the package Biobase is
        'AnnotatedDataFrame-class'.
        