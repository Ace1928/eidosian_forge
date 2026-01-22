import re
from jedi.inference.names import AbstractArbitraryName
from jedi.inference.helpers import infer_call_of_leaf
from jedi.api.classes import Completion
from jedi.parser_utils import cut_value_at_position
def complete_dict(module_context, code_lines, leaf, position, string, fuzzy):
    bracket_leaf = leaf
    if bracket_leaf != '[':
        bracket_leaf = leaf.get_previous_leaf()
    cut_end_quote = ''
    if string:
        cut_end_quote = get_quote_ending(string, code_lines, position, invert_result=True)
    if bracket_leaf == '[':
        if string is None and leaf is not bracket_leaf:
            string = cut_value_at_position(leaf, position)
        context = module_context.create_context(bracket_leaf)
        before_node = before_bracket_leaf = bracket_leaf.get_previous_leaf()
        if before_node in (')', ']', '}'):
            before_node = before_node.parent
        if before_node.type in ('atom', 'trailer', 'name'):
            values = infer_call_of_leaf(context, before_bracket_leaf)
            return list(_completions_for_dicts(module_context.inference_state, values, '' if string is None else string, cut_end_quote, fuzzy=fuzzy))
    return []