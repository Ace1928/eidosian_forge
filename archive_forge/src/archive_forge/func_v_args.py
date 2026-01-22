from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
def v_args(inline: bool=False, meta: bool=False, tree: bool=False, wrapper: Optional[Callable]=None) -> Callable[[_DECORATED], _DECORATED]:
    """A convenience decorator factory for modifying the behavior of user-supplied visitor methods.

    By default, callback methods of transformers/visitors accept one argument - a list of the node's children.

    ``v_args`` can modify this behavior. When used on a transformer/visitor class definition,
    it applies to all the callback methods inside it.

    ``v_args`` can be applied to a single method, or to an entire class. When applied to both,
    the options given to the method take precedence.

    Parameters:
        inline (bool, optional): Children are provided as ``*args`` instead of a list argument (not recommended for very long lists).
        meta (bool, optional): Provides two arguments: ``meta`` and ``children`` (instead of just the latter)
        tree (bool, optional): Provides the entire tree as the argument, instead of the children.
        wrapper (function, optional): Provide a function to decorate all methods.

    Example:
        ::

            @v_args(inline=True)
            class SolveArith(Transformer):
                def add(self, left, right):
                    return left + right

                @v_args(meta=True)
                def mul(self, meta, children):
                    logger.info(f'mul at line {meta.line}')
                    left, right = children
                    return left * right


            class ReverseNotation(Transformer_InPlace):
                @v_args(tree=True)
                def tree_node(self, tree):
                    tree.children = tree.children[::-1]
    """
    if tree and (meta or inline):
        raise ValueError("Visitor functions cannot combine 'tree' with 'meta' or 'inline'.")
    func = None
    if meta:
        if inline:
            func = _vargs_meta_inline
        else:
            func = _vargs_meta
    elif inline:
        func = _vargs_inline
    elif tree:
        func = _vargs_tree
    if wrapper is not None:
        if func is not None:
            raise ValueError("Cannot use 'wrapper' along with 'tree', 'meta' or 'inline'.")
        func = wrapper

    def _visitor_args_dec(obj):
        return _apply_v_args(obj, func)
    return _visitor_args_dec