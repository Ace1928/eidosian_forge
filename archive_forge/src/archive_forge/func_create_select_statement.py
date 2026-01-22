import sys
import numpy as np
from pymysql import connect
from pymysql.err import ProgrammingError
from copy import deepcopy
from ase.db.sqlite import SQLite3Database
from ase.db.sqlite import init_statements
from ase.db.sqlite import VERSION
from ase.db.postgresql import remove_nan_and_inf, insert_nan_and_inf
import ase.io.jsonio
import json
def create_select_statement(self, keys, cmps, sort=None, order=None, sort_table=None, what='systems.*'):
    sql, value = super(MySQLDatabase, self).create_select_statement(keys, cmps, sort, order, sort_table, what)
    for subst in MySQLCursor.sql_replace:
        sql = sql.replace(subst[0], subst[1])
    return (sql, value)