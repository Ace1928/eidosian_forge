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
def visit_CStructOrUnionDefNode(self, node):
    if True:
        return None
    self_value = ExprNodes.AttributeNode(pos=node.pos, obj=ExprNodes.NameNode(pos=node.pos, name=u'self'), attribute=EncodedString(u'value'))
    var_entries = node.entry.type.scope.var_entries
    attributes = []
    for entry in var_entries:
        attributes.append(ExprNodes.AttributeNode(pos=entry.pos, obj=self_value, attribute=entry.name))
    init_assignments = []
    for entry, attr in zip(var_entries, attributes):
        init_assignments.append(self.init_assignment.substitute({u'VALUE': ExprNodes.NameNode(entry.pos, name=entry.name), u'ATTR': attr}, pos=entry.pos))
    str_format = u'%s(%s)' % (node.entry.type.name, ('%s, ' * len(attributes))[:-2])
    wrapper_class = self.struct_or_union_wrapper.substitute({u'INIT_ASSIGNMENTS': Nodes.StatListNode(node.pos, stats=init_assignments), u'IS_UNION': ExprNodes.BoolNode(node.pos, value=not node.entry.type.is_struct), u'MEMBER_TUPLE': ExprNodes.TupleNode(node.pos, args=attributes), u'STR_FORMAT': ExprNodes.StringNode(node.pos, value=EncodedString(str_format)), u'REPR_FORMAT': ExprNodes.StringNode(node.pos, value=EncodedString(str_format.replace('%s', '%r')))}, pos=node.pos).stats[0]
    wrapper_class.class_name = node.name
    wrapper_class.shadow = True
    class_body = wrapper_class.body.stats
    assert isinstance(class_body[0].base_type, Nodes.CSimpleBaseTypeNode)
    class_body[0].base_type.name = node.name
    init_method = class_body[1]
    assert isinstance(init_method, Nodes.DefNode) and init_method.name == '__init__'
    arg_template = init_method.args[1]
    if not node.entry.type.is_struct:
        arg_template.kw_only = True
    del init_method.args[1]
    for entry, attr in zip(var_entries, attributes):
        arg = copy.deepcopy(arg_template)
        arg.declarator.name = entry.name
        init_method.args.append(arg)
    for entry, attr in zip(var_entries, attributes):
        if entry.type.is_pyobject:
            template = self.basic_pyobject_property
        else:
            template = self.basic_property
        property = template.substitute({u'ATTR': attr}, pos=entry.pos).stats[0]
        property.name = entry.name
        wrapper_class.body.stats.append(property)
    wrapper_class.analyse_declarations(self.current_env())
    return self.visit_CClassDefNode(wrapper_class)