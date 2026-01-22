from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
def visit_CImportStatNode(self, node):
    module_name = node.module_name
    if module_name == u'cython.cimports':
        error(node.pos, "Cannot cimport the 'cython.cimports' package directly, only submodules.")
    if module_name.startswith(u'cython.cimports.'):
        if node.as_name and node.as_name != u'cython':
            node.module_name = module_name[len(u'cython.cimports.'):]
            return node
        error(node.pos, "Python cimports must use 'from cython.cimports... import ...' or 'import ... as ...', not just 'import ...'")
    if module_name == u'cython':
        self.cython_module_names.add(node.as_name or u'cython')
    elif module_name.startswith(u'cython.'):
        if module_name.startswith(u'cython.parallel.'):
            error(node.pos, node.module_name + ' is not a module')
        else:
            self._check_valid_cython_module(node.pos, module_name)
        if module_name == u'cython.parallel':
            if node.as_name and node.as_name != u'cython':
                self.parallel_directives[node.as_name] = module_name
            else:
                self.cython_module_names.add(u'cython')
                self.parallel_directives[u'cython.parallel'] = module_name
            self.module_scope.use_utility_code(UtilityCode.load_cached('InitThreads', 'ModuleSetupCode.c'))
        elif node.as_name:
            self.directive_names[node.as_name] = module_name[7:]
        else:
            self.cython_module_names.add(u'cython')
        return None
    return node