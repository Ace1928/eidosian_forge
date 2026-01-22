from sympy.physics.quantum.cartesian import (XOp, YOp, ZOp, XKet, PxOp, PxKet,
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import StateBase, BraBase, Ket
from sympy.physics.quantum.spin import (JxOp, JyOp, JzOp, J2Op, JxKet, JyKet,
def operators_to_state(operators, **options):
    """ Returns the eigenstate of the given operator or set of operators

    A global function for mapping operator classes to their associated
    states. It takes either an Operator or a set of operators and
    returns the state associated with these.

    This function can handle both instances of a given operator or
    just the class itself (i.e. both XOp() and XOp)

    There are multiple use cases to consider:

    1) A class or set of classes is passed: First, we try to
    instantiate default instances for these operators. If this fails,
    then the class is simply returned. If we succeed in instantiating
    default instances, then we try to call state._operators_to_state
    on the operator instances. If this fails, the class is returned.
    Otherwise, the instance returned by _operators_to_state is returned.

    2) An instance or set of instances is passed: In this case,
    state._operators_to_state is called on the instances passed. If
    this fails, a state class is returned. If the method returns an
    instance, that instance is returned.

    In both cases, if the operator class or set does not exist in the
    state_mapping dictionary, None is returned.

    Parameters
    ==========

    arg: Operator or set
         The class or instance of the operator or set of operators
         to be mapped to a state

    Examples
    ========

    >>> from sympy.physics.quantum.cartesian import XOp, PxOp
    >>> from sympy.physics.quantum.operatorset import operators_to_state
    >>> from sympy.physics.quantum.operator import Operator
    >>> operators_to_state(XOp)
    |x>
    >>> operators_to_state(XOp())
    |x>
    >>> operators_to_state(PxOp)
    |px>
    >>> operators_to_state(PxOp())
    |px>
    >>> operators_to_state(Operator)
    |psi>
    >>> operators_to_state(Operator())
    |psi>
    """
    if not (isinstance(operators, Operator) or isinstance(operators, set) or issubclass(operators, Operator)):
        raise NotImplementedError('Argument is not an Operator or a set!')
    if isinstance(operators, set):
        for s in operators:
            if not (isinstance(s, Operator) or issubclass(s, Operator)):
                raise NotImplementedError('Set is not all Operators!')
        ops = frozenset(operators)
        if ops in op_mapping:
            try:
                op_instances = [op() for op in ops]
                ret = _get_state(op_mapping[ops], set(op_instances), **options)
            except NotImplementedError:
                ret = op_mapping[ops]
            return ret
        else:
            tmp = [type(o) for o in ops]
            classes = frozenset(tmp)
            if classes in op_mapping:
                ret = _get_state(op_mapping[classes], ops, **options)
            else:
                ret = None
            return ret
    elif operators in op_mapping:
        try:
            op_instance = operators()
            ret = _get_state(op_mapping[operators], op_instance, **options)
        except NotImplementedError:
            ret = op_mapping[operators]
        return ret
    elif type(operators) in op_mapping:
        return _get_state(op_mapping[type(operators)], operators, **options)
    else:
        return None