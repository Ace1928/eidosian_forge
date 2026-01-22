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
def search_bm25f(cls, term, weights=None, with_score=False, score_alias='score', explicit_ordering=False):
    """Full-text search for selected `term` using BM25 algorithm."""
    return cls._search(term, weights, with_score, score_alias, cls.bm25f, explicit_ordering)