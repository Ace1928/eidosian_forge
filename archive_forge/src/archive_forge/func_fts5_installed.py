import json
import math
import re
import struct
import sys
from peewee import *
from peewee import ColumnBase
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Insert
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import VirtualField
from peewee import merge_dict
from peewee import sqlite3
@classmethod
def fts5_installed(cls):
    if sqlite3.sqlite_version_info[:3] < FTS5_MIN_SQLITE_VERSION:
        return False
    tmp_db = sqlite3.connect(':memory:')
    try:
        tmp_db.execute('CREATE VIRTUAL TABLE fts5test USING fts5 (data);')
    except:
        try:
            tmp_db.enable_load_extension(True)
            tmp_db.load_extension('fts5')
        except:
            return False
        else:
            cls._meta.database.load_extension('fts5')
    finally:
        tmp_db.close()
    return True