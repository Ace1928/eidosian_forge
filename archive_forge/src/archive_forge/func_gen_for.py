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
def gen_for(self, node, target, local_iter, local_iter_decl, loop_body):
    """
        Create For representation on iterator for Cxx generation.

        Examples
        --------
        >> "omp parallel for"
        >> for i in range(10):
        >>     ... do things ...

        Becomes

        >> "omp parallel for shared(__iterX)"
        >> for(decltype(__iterX)::iterator __targetX = __iterX.begin();
               __targetX < __iterX.end(); ++__targetX)
        >>         auto&& i = *__targetX;
        >>     ... do things ...

        It the case of not local variable, typing for `i` disappear and typing
        is removed for iterator in case of yields statement in function.
        """
    local_target = '__target{0}'.format(id(node))
    local_target_decl = self.typeof(self.types.builder.IteratorOfType(local_iter_decl))
    islocal = node.target.id not in self.openmp_deps and node.target.id in self.scope[node] and (not hasattr(self, 'yields'))
    if islocal:
        local_type = 'auto&&'
        self.ldecls.remove(node.target.id)
    else:
        local_type = ''
    loop_body_prelude = Statement('{} {}= *{}'.format(local_type, target, local_target))
    assign = self.make_assign(local_target_decl, local_target, local_iter)
    loop = For('{}.begin()'.format(assign), '{0} < {1}.end()'.format(local_target, local_iter), '++{0}'.format(local_target), Block([loop_body_prelude, loop_body]))
    return [self.process_omp_attachements(node, loop)]