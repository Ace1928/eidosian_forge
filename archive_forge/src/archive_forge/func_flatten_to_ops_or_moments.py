from typing import Callable, Iterable, Iterator, NoReturn, Union, TYPE_CHECKING
from typing_extensions import Protocol
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops.raw_types import Operation
def flatten_to_ops_or_moments(root: OP_TREE) -> Iterator[Union[Operation, 'cirq.Moment']]:
    """Performs an in-order iteration OP_TREE, yielding ops and moments.

    Args:
        root: The operation or tree of operations to iterate.

    Yields:
        Operations or moments from the tree.

    Raises:
        TypeError: root isn't a valid OP_TREE.
    """
    if isinstance(root, (Operation, moment.Moment)):
        yield root
    elif isinstance(root, Iterable) and (not isinstance(root, str)):
        for subtree in root:
            yield from flatten_to_ops_or_moments(subtree)
    else:
        _bad_op_tree(root)