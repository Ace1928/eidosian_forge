from collections import deque
from sympy.core.random import randint
from sympy.external import import_module
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.numbers import Number, equal_valued
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.dagger import Dagger
def process_new_rule(new_rule, ops):
    if new_rule is not None:
        new_left, new_right = new_rule
        if new_rule not in rules and (new_right, new_left) not in rules:
            rules.add(new_rule)
        if ops + 1 < max_ops:
            queue.append(new_rule + (ops + 1,))