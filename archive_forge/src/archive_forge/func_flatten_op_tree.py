from typing import Callable, Iterable, Iterator, NoReturn, Union, TYPE_CHECKING
from typing_extensions import Protocol
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops.raw_types import Operation
def flatten_op_tree(root: OP_TREE, preserve_moments: bool=False) -> Iterator[Union[Operation, 'cirq.Moment']]:
    """Performs an in-order iteration of the operations (leaves) in an OP_TREE.

    Args:
        root: The operation or tree of operations to iterate.
        preserve_moments: Whether to yield Moments intact instead of
            flattening them

    Yields:
        Operations from the tree.

    Raises:
        TypeError: root isn't a valid OP_TREE.
    """
    if preserve_moments:
        return flatten_to_ops_or_moments(root)
    else:
        return flatten_to_ops(root)