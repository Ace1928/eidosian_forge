from pythran.analyses import LocalNodeDeclarations, GlobalDeclarations, Scope
from pythran.analyses import YieldPoints, IsAssigned, ASTMatcher, AST_any
from pythran.analyses import RangeValues, PureExpressions, Dependencies
from pythran.analyses import Immediates, Ancestors, StrictAliases
from pythran.config import cfg
from pythran.cxxgen import Template, Include, Namespace, CompilationUnit
from pythran.cxxgen import Statement, Block, AnnotatedStatement, Typedef, Label
from pythran.cxxgen import Value, FunctionDeclaration, EmptyStatement, Nop
from pythran.cxxgen import FunctionBody, Line, ReturnStatement, Struct, Assign
from pythran.cxxgen import For, While, TryExcept, ExceptHandler, If, AutoFor
from pythran.cxxgen import StatementWithComments
from pythran.openmp import OMPDirective
from pythran.passmanager import Backend
from pythran.syntax import PythranSyntaxError
from pythran.tables import operator_to_lambda, update_operator_to_lambda
from pythran.tables import pythran_ward, attributes as attributes_table
from pythran.types.conversion import PYTYPE_TO_CTYPE_TABLE, TYPE_TO_SUFFIX
from pythran.types.types import Types
from pythran.utils import attr_to_path, pushpop, cxxid, isstr, isnum
from pythran.utils import isextslice, ispowi, quote_cxxstring
from pythran import metadata, unparse
from math import isnan, isinf
import gast as ast
from functools import reduce
import io
def handle_omp_for(self, node, local_iter):
    """
        Fix OpenMP directives on For loops.

        Add the target as private variable as a new variable may have been
        introduce to handle cxx iterator.

        Also, add the iterator as shared variable as all 'parallel for chunck'
        have to use the same iterator.
        """
    for directive in metadata.get(node, OMPDirective):
        if any((key in directive.s for key in (' parallel ', ' task '))):
            directive.s += ' shared({})'
            directive.deps.append(ast.Name(local_iter, ast.Load(), None, None))
            directive.shared_deps.append(directive.deps[-1])
        target = node.target
        assert isinstance(target, ast.Name)
        hasfor = 'for' in directive.s
        nodefault = 'default' not in directive.s
        noindexref = all((isinstance(x, ast.Name) and x.id != target.id for x in directive.deps))
        if hasfor and nodefault and noindexref and (target.id not in self.scope[node]):
            directive.s += ' private({})'
            directive.deps.append(ast.Name(target.id, ast.Load(), None, None))
            directive.private_deps.append(directive.deps[-1])