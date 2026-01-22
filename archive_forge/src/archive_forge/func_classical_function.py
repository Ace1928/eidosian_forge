from .classicalfunction import ClassicalFunction
from .exceptions import (
from .boolean_expression import BooleanExpression
def classical_function(func):
    """
    Parses and type checks the callable ``func`` to compile it into an ``ClassicalFunction``
    that can be synthesized into a ``QuantumCircuit``.

    Args:
        func (callable): A callable (with type hints) to compile into an ``ClassicalFunction``.

    Returns:
        ClassicalFunction: An object that can synthesis into a QuantumCircuit (via ``synth()``
        method).
    """
    import inspect
    from textwrap import dedent
    source = dedent(inspect.getsource(func))
    return ClassicalFunction(source, name=func.__name__)