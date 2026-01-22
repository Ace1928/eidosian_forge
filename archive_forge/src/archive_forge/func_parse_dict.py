import html.entities as htmlentitydefs
import re
import warnings
from ast import literal_eval
from collections import defaultdict
from enum import Enum
from io import StringIO
from typing import Any, NamedTuple
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file
def parse_dict(curr_token):
    curr_token = consume(curr_token, Pattern.DICT_START, "'['")
    curr_token, dct = parse_kv(curr_token)
    curr_token = consume(curr_token, Pattern.DICT_END, "']'")
    return (curr_token, dct)