from xdg.Menu import parse, Menu, MenuEntry
import os
import locale
import subprocess
import ast
import sys
from xdg.BaseDirectory import xdg_data_dirs, xdg_config_dirs
from xdg.DesktopEntry import DesktopEntry
from xdg.Exceptions import ParsingError
from xdg.util import PY3
import xdg.Locale
import xdg.Config
@classmethod
def fromFilename(cls, type, filename):
    tree = ast.Expression(body=ast.Compare(left=ast.Str(filename), ops=[ast.Eq()], comparators=[ast.Attribute(value=ast.Name(id='menuentry', ctx=ast.Load()), attr='DesktopFileID', ctx=ast.Load())]), lineno=1, col_offset=0)
    ast.fix_missing_locations(tree)
    rule = Rule(type, tree)
    return rule