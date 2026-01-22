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
@property
def fts_column_index(self):
    if not hasattr(self, '_fts_column_index'):
        search_fields = [f.name for f in self.model._meta.sorted_fields if isinstance(f, SearchField)]
        self._fts_column_index = search_fields.index(self.name)
    return self._fts_column_index