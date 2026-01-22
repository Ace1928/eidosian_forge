import copy
import itertools
from parso.python import tree
from jedi import debug
from jedi import parser_utils
from jedi.inference.base_value import ValueSet, NO_VALUES, ContextualizedNode, \
from jedi.inference.lazy_value import LazyTreeValue
from jedi.inference import compiled
from jedi.inference import recursion
from jedi.inference import analysis
from jedi.inference import imports
from jedi.inference import arguments
from jedi.inference.value import ClassValue, FunctionValue
from jedi.inference.value import iterable
from jedi.inference.value.dynamic_arrays import ListModification, DictModification
from jedi.inference.value import TreeInstance
from jedi.inference.helpers import is_string, is_literal, is_number, \
from jedi.inference.compiled.access import COMPARISON_OPERATORS
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.gradual.stub_value import VersionInfo
from jedi.inference.gradual import annotation
from jedi.inference.names import TreeNameDefinition
from jedi.inference.context import CompForContext
from jedi.inference.value.decorator import Decoratee
from jedi.plugins import plugin_manager
def infer_atom(context, atom):
    """
    Basically to process ``atom`` nodes. The parser sometimes doesn't
    generate the node (because it has just one child). In that case an atom
    might be a name or a literal as well.
    """
    state = context.inference_state
    if atom.type == 'name':
        stmt = tree.search_ancestor(atom, 'expr_stmt', 'lambdef', 'if_stmt') or atom
        if stmt.type == 'if_stmt':
            if not any((n.start_pos <= atom.start_pos < n.end_pos for n in stmt.get_test_nodes())):
                stmt = atom
        elif stmt.type == 'lambdef':
            stmt = atom
        position = stmt.start_pos
        if _is_annotation_name(atom):
            position = None
        return context.py__getattribute__(atom, position=position)
    elif atom.type == 'keyword':
        if atom.value in ('False', 'True', 'None'):
            return ValueSet([compiled.builtin_from_name(state, atom.value)])
        elif atom.value == 'yield':
            return NO_VALUES
        assert False, 'Cannot infer the keyword %s' % atom
    elif isinstance(atom, tree.Literal):
        string = state.compiled_subprocess.safe_literal_eval(atom.value)
        return ValueSet([compiled.create_simple_object(state, string)])
    elif atom.type == 'strings':
        value_set = infer_atom(context, atom.children[0])
        for string in atom.children[1:]:
            right = infer_atom(context, string)
            value_set = _infer_comparison(context, value_set, '+', right)
        return value_set
    elif atom.type == 'fstring':
        return compiled.get_string_value_set(state)
    else:
        c = atom.children
        if c[0] == '(' and (not len(c) == 2) and (not (c[1].type == 'testlist_comp' and len(c[1].children) > 1)):
            return context.infer_node(c[1])
        try:
            comp_for = c[1].children[1]
        except (IndexError, AttributeError):
            pass
        else:
            if comp_for == ':':
                try:
                    comp_for = c[1].children[3]
                except IndexError:
                    pass
            if comp_for.type in ('comp_for', 'sync_comp_for'):
                return ValueSet([iterable.comprehension_from_atom(state, context, atom)])
        array_node = c[1]
        try:
            array_node_c = array_node.children
        except AttributeError:
            array_node_c = []
        if c[0] == '{' and (array_node == '}' or ':' in array_node_c or '**' in array_node_c):
            new_value = iterable.DictLiteralValue(state, context, atom)
        else:
            new_value = iterable.SequenceLiteralValue(state, context, atom)
        return ValueSet([new_value])