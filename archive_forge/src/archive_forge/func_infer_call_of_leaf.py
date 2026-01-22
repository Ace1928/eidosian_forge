import copy
import sys
import re
import os
from itertools import chain
from contextlib import contextmanager
from parso.python import tree
def infer_call_of_leaf(context, leaf, cut_own_trailer=False):
    """
    Creates a "call" node that consist of all ``trailer`` and ``power``
    objects.  E.g. if you call it with ``append``::

        list([]).append(3) or None

    You would get a node with the content ``list([]).append`` back.

    This generates a copy of the original ast node.

    If you're using the leaf, e.g. the bracket `)` it will return ``list([])``.

    We use this function for two purposes. Given an expression ``bar.foo``,
    we may want to
      - infer the type of ``foo`` to offer completions after foo
      - infer the type of ``bar`` to be able to jump to the definition of foo
    The option ``cut_own_trailer`` must be set to true for the second purpose.
    """
    trailer = leaf.parent
    if trailer.type == 'fstring':
        from jedi.inference import compiled
        return compiled.get_string_value_set(context.inference_state)
    if trailer.type != 'trailer' or leaf not in (trailer.children[0], trailer.children[-1]):
        if leaf == ':':
            from jedi.inference.base_value import NO_VALUES
            return NO_VALUES
        if trailer.type == 'atom':
            return context.infer_node(trailer)
        return context.infer_node(leaf)
    power = trailer.parent
    index = power.children.index(trailer)
    if cut_own_trailer:
        cut = index
    else:
        cut = index + 1
    if power.type == 'error_node':
        start = index
        while True:
            start -= 1
            base = power.children[start]
            if base.type != 'trailer':
                break
        trailers = power.children[start + 1:cut]
    else:
        base = power.children[0]
        trailers = power.children[1:cut]
    if base == 'await':
        base = trailers[0]
        trailers = trailers[1:]
    values = context.infer_node(base)
    from jedi.inference.syntax_tree import infer_trailer
    for trailer in trailers:
        values = infer_trailer(context, values, trailer)
    return values