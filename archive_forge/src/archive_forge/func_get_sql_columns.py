from typing import Optional, List
import numpy as np
from ase.db.core import float_to_time_string, now
def get_sql_columns(columns):
    """ Map the names of table columns to names of columns in
    the SQL tables"""
    sql_columns = columns[:]
    if 'age' in columns:
        sql_columns.remove('age')
        sql_columns += ['mtime', 'ctime']
    if 'user' in columns:
        sql_columns[sql_columns.index('user')] = 'username'
    if 'formula' in columns:
        sql_columns[sql_columns.index('formula')] = 'numbers'
    if 'fmax' in columns:
        sql_columns[sql_columns.index('fmax')] = 'forces'
    if 'smax' in columns:
        sql_columns[sql_columns.index('smax')] = 'stress'
    if 'volume' in columns:
        sql_columns[sql_columns.index('volume')] = 'cell'
    if 'mass' in columns:
        sql_columns[sql_columns.index('mass')] = 'masses'
    if 'charge' in columns:
        sql_columns[sql_columns.index('charge')] = 'charges'
    sql_columns.append('key_value_pairs')
    sql_columns.append('constraints')
    if 'id' not in sql_columns:
        sql_columns.append('id')
    return sql_columns