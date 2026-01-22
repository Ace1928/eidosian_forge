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
def process_omp_attachements(self, node, stmt, index=None):
    """
        Add OpenMP pragma on the correct stmt in the correct order.

        stmt may be a list. On this case, index have to be specify to add
        OpenMP on the correct statement.
        """
    omp_directives = metadata.get(node, OMPDirective)
    if omp_directives:
        directives = list()
        for directive in omp_directives:
            directive.deps = [self.visit(dep) for dep in directive.deps]
            directives.append(directive)
        if index is None:
            stmt = AnnotatedStatement(stmt, directives)
        else:
            stmt[index] = AnnotatedStatement(stmt[index], directives)
    return stmt