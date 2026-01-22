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
def gen_c_for(self, node, local_iter, loop_body):
    """
        Create C For representation for Cxx generation.

        Examples
        --------
        >> for i in range(10):
        >>     ... do things ...

        Becomes

        >> for(long i = 0, __targetX = 10; i < __targetX; i += 1)
        >>     ... do things ...

        Or

        >> for i in range(10, 0, -1):
        >>     ... do things ...

        Becomes

        >> for(long i = 10, __targetX = 0; i > __targetX; i += -1)
        >>     ... do things ...


        It the case of not local variable, typing for `i` disappear
        """
    args = node.iter.args
    step = '1L' if len(args) <= 2 else self.visit(args[2])
    if len(args) == 1:
        lower_bound = '0L'
        upper_arg = 0
    else:
        lower_bound = self.visit(args[0])
        upper_arg = 1
    upper_type = iter_type = 'long '
    upper_value = self.visit(args[upper_arg])
    if self.is_in_collapse(node, args[upper_arg]):
        upper_bound = upper_value
    else:
        upper_bound = '__target{0}'.format(id(node))
    islocal = node.target.id not in self.openmp_deps and node.target.id in self.scope[node] and (not hasattr(self, 'yields'))
    if islocal:
        loop = list()
        self.ldecls.remove(node.target.id)
    else:
        iter_type = ''
        if node.target.id in self.scope[node]:
            loop = []
        else:
            loop = [If('{} == {}'.format(local_iter, upper_bound), Statement('{} -= {}'.format(local_iter, step)))]
    comparison = self.handle_real_loop_comparison(args, local_iter, upper_bound)
    forloop = For('{0} {1}={2}'.format(iter_type, local_iter, lower_bound), comparison, '{0} += {1}'.format(local_iter, step), loop_body)
    loop.insert(0, self.process_omp_attachements(node, forloop))
    if upper_bound is upper_value:
        header = []
    else:
        assgnt = self.make_assign(upper_type, upper_bound, upper_value)
        header = [Statement(assgnt)]
    return (header, loop)