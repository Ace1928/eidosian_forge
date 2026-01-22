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
def bm25f(cls, *weights):
    match_info = fn.matchinfo(cls._meta.entity, FTS4_MATCHINFO)
    return fn.fts_bm25f(match_info, *weights)