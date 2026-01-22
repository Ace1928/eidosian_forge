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
def create_metaRd_db(dbcon) -> None:
    """ Create an database to store R help pages.

    dbcon: database connection (assumed to be SQLite - may or may not work
           with other databases)
    """
    dbcon.execute('\nCREATE TABLE package (\nname TEXT UNIQUE,\ntitle TEXT,\nversion TEXT,\ndescription TEXT\n);\n')
    dbcon.execute('\nCREATE TABLE rd_meta (\nid INTEGER, file TEXT UNIQUE, name TEXT, type TEXT, title TEXT, encoding TEXT,\npackage_rowid INTEGER\n);\n')
    dbcon.execute('\nCREATE INDEX type_idx ON rd_meta (type);\n')
    dbcon.execute('\nCREATE TABLE rd_alias_meta (\nrd_meta_rowid INTEGER, alias TEXT\n);\n')
    dbcon.execute('\nCREATE INDEX alias_idx ON rd_alias_meta (alias);\n')
    dbcon.commit()