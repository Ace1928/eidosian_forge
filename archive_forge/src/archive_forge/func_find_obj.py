import builtins
import inspect
import re
import sys
import typing
import warnings
from inspect import Parameter
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Type, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import Inliner
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref, pending_xref_condition
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, Index, IndexEntry, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.pycode.ast import ast
from sphinx.pycode.ast import parse as ast_parse
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective, switch_source_input
from sphinx.util.inspect import signature_from_str
from sphinx.util.nodes import (find_pending_xref_condition, make_id, make_refnode,
from sphinx.util.typing import OptionSpec, TextlikeNode
def find_obj(self, env: BuildEnvironment, modname: str, classname: str, name: str, type: str, searchmode: int=0) -> List[Tuple[str, ObjectEntry]]:
    """Find a Python object for "name", perhaps using the given module
        and/or classname.  Returns a list of (name, object entry) tuples.
        """
    if name[-2:] == '()':
        name = name[:-2]
    if not name:
        return []
    matches: List[Tuple[str, ObjectEntry]] = []
    newname = None
    if searchmode == 1:
        if type is None:
            objtypes = list(self.object_types)
        else:
            objtypes = self.objtypes_for_role(type)
        if objtypes is not None:
            if modname and classname:
                fullname = modname + '.' + classname + '.' + name
                if fullname in self.objects and self.objects[fullname].objtype in objtypes:
                    newname = fullname
            if not newname:
                if modname and modname + '.' + name in self.objects and (self.objects[modname + '.' + name].objtype in objtypes):
                    newname = modname + '.' + name
                elif name in self.objects and self.objects[name].objtype in objtypes:
                    newname = name
                else:
                    searchname = '.' + name
                    matches = [(oname, self.objects[oname]) for oname in self.objects if oname.endswith(searchname) and self.objects[oname].objtype in objtypes]
    elif name in self.objects:
        newname = name
    elif type == 'mod':
        return []
    elif classname and classname + '.' + name in self.objects:
        newname = classname + '.' + name
    elif modname and modname + '.' + name in self.objects:
        newname = modname + '.' + name
    elif modname and classname and (modname + '.' + classname + '.' + name in self.objects):
        newname = modname + '.' + classname + '.' + name
    if newname is not None:
        matches.append((newname, self.objects[newname]))
    return matches