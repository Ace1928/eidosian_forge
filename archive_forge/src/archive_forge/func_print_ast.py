import collections
import collections.abc
import logging
import sys
import textwrap
from abc import ABC
def print_ast(ast_tree, *, end_marker_after=5, output_function=None):
    """Debugging aid, which can dump a Deb822Element or a list of tokens/elements

    :param ast_tree: Either a Deb822Element or an iterable Deb822Token/Deb822Element entries
      (both types may be mixed in the same iterable, which enable it to dump the
      ast tree at different stages of parse_deb822_file method)
    :param end_marker_after: The dump will add "end of element" markers if a
      given element spans at least this many tokens/elements. Can be disabled
      with by passing None as value. Use 0 for unconditionally marking all
      elements (note that tokens never get an "end of element" marker as they
      are not an elements).
    :param output_function: Callable that receives a single str argument and is responsible
      for "displaying" that line. The callable may be invoked multiple times (one per line
      of output).  Defaults to logging.info if omitted.

    """
    from debian._deb822_repro.parsing import Deb822Element
    prefix = None
    if isinstance(ast_tree, Deb822Element):
        ast_tree = [ast_tree]
    stack = [(0, '', iter(ast_tree))]
    current_no = 0
    if output_function is None:
        output_function = logging.info
    while stack:
        start_no, name, current_iter = stack[-1]
        for current in current_iter:
            current_no += 1
            if prefix is None:
                prefix = '  ' * len(stack)
            if isinstance(current, Deb822Element):
                stack.append((current_no, current.__class__.__name__, iter(current.iter_parts())))
                output_function(prefix + current.__class__.__name__)
                prefix = None
                break
            output_function(prefix + str(current))
        else:
            stack.pop()
            prefix = None
            if end_marker_after is not None and start_no + end_marker_after <= current_no and name:
                if prefix is None:
                    prefix = '  ' * len(stack)
                output_function(prefix + '# <-- END OF ' + name)