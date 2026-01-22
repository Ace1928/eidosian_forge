from typing import (
import cmath
import re
import numpy as np
import sympy
def feed_op(could_be_binary: bool, token: Any) -> None:
    mul = cast(_CustomQuirkOperationToken, token_map['*'])
    if could_be_binary and token != ')':
        if not isinstance(token, _CustomQuirkOperationToken) or token.binary_action is None:
            burn_ops(mul.priority)
            ops.append(_HangingNode(func=cast(Callable[[T, T], T], mul.binary_action), weight=mul.priority))
    if isinstance(token, _CustomQuirkOperationToken):
        if could_be_binary and token.binary_action is not None:
            burn_ops(token.priority)
            ops.append(_HangingNode(func=token.binary_action, weight=token.priority))
        elif token.unary_action is not None:
            burn_ops(token.priority)
            vals.append(None)
            ops.append(_HangingNode(func=lambda _, b: token.unary_action(b), weight=np.inf))
        elif token.binary_action is not None:
            raise ValueError('Bad expression: binary op in bad spot.\ntext={text!r}')