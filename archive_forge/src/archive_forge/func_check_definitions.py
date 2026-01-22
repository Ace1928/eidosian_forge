from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def check_definitions(flow, compiler_directives):
    flow.initialize()
    flow.reaching_definitions()
    assignments = set()
    references = {}
    assmt_nodes = set()
    for block in flow.blocks:
        i_state = block.i_input
        for stat in block.stats:
            i_assmts = flow.assmts[stat.entry]
            state = flow.map_one(i_state, stat.entry)
            if isinstance(stat, NameAssignment):
                stat.lhs.cf_state.update(state)
                assmt_nodes.add(stat.lhs)
                i_state = i_state & ~i_assmts.mask
                if stat.is_deletion:
                    i_state |= i_assmts.bit
                else:
                    i_state |= stat.bit
                assignments.add(stat)
                if stat.rhs is not fake_rhs_expr:
                    stat.entry.cf_assignments.append(stat)
            elif isinstance(stat, NameReference):
                references[stat.node] = stat.entry
                stat.entry.cf_references.append(stat)
                stat.node.cf_state.update(state)
                state.discard(Uninitialized)
                state.discard(Unknown)
                for assmt in state:
                    assmt.refs.add(stat)
    warn_maybe_uninitialized = compiler_directives['warn.maybe_uninitialized']
    warn_unused_result = compiler_directives['warn.unused_result']
    warn_unused = compiler_directives['warn.unused']
    warn_unused_arg = compiler_directives['warn.unused_arg']
    messages = MessageCollection()
    for node in assmt_nodes:
        if Uninitialized in node.cf_state:
            node.cf_maybe_null = True
            if len(node.cf_state) == 1:
                node.cf_is_null = True
            else:
                node.cf_is_null = False
        elif Unknown in node.cf_state:
            node.cf_maybe_null = True
        else:
            node.cf_is_null = False
            node.cf_maybe_null = False
    for node, entry in references.items():
        if Uninitialized in node.cf_state:
            node.cf_maybe_null = True
            if not entry.from_closure and len(node.cf_state) == 1 and (entry.name not in entry.scope.scope_predefined_names):
                node.cf_is_null = True
            if node.allow_null or entry.from_closure or entry.is_pyclass_attr or entry.type.is_error:
                pass
            elif node.cf_is_null and (not entry.in_closure):
                if entry.error_on_uninitialized or (Options.error_on_uninitialized and (entry.type.is_pyobject or entry.type.is_unspecified)):
                    messages.error(node.pos, "local variable '%s' referenced before assignment" % entry.name)
                else:
                    messages.warning(node.pos, "local variable '%s' referenced before assignment" % entry.name)
            elif warn_maybe_uninitialized:
                msg = "local variable '%s' might be referenced before assignment" % entry.name
                if entry.in_closure:
                    msg += ' (maybe initialized inside a closure)'
                messages.warning(node.pos, msg)
        elif Unknown in node.cf_state:
            node.cf_maybe_null = True
        else:
            node.cf_is_null = False
            node.cf_maybe_null = False
    for assmt in assignments:
        if not assmt.refs and (not assmt.entry.is_pyclass_attr) and (not assmt.entry.in_closure):
            if assmt.entry.cf_references and warn_unused_result:
                if assmt.is_arg:
                    messages.warning(assmt.pos, "Unused argument value '%s'" % assmt.entry.name)
                else:
                    messages.warning(assmt.pos, "Unused result in '%s'" % assmt.entry.name)
            assmt.lhs.cf_used = False
    for entry in flow.entries:
        if not entry.cf_references and (not entry.is_pyclass_attr):
            if entry.name != '_' and (not entry.name.startswith('unused')):
                if entry.is_arg:
                    if warn_unused_arg:
                        messages.warning(entry.pos, "Unused argument '%s'" % entry.name)
                elif warn_unused:
                    messages.warning(entry.pos, "Unused entry '%s'" % entry.name)
            entry.cf_used = False
    messages.report()
    for node in assmt_nodes:
        node.cf_state = ControlFlowState(node.cf_state)
    for node in references:
        node.cf_state = ControlFlowState(node.cf_state)