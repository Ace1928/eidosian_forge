from cirq import ops
def assert_equivalent_op_tree(x: ops.OP_TREE, y: ops.OP_TREE):
    """Ensures that the two OP_TREEs are equivalent.

    Args:
        x: OP_TREE one
        y: OP_TREE two
    Returns:
        None
    Raises:
         AssertionError if x != y
    """
    a = list(ops.flatten_op_tree(x))
    b = list(ops.flatten_op_tree(y))
    assert a == b