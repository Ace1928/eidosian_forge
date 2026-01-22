from __future__ import annotations
import abc
import typing
@property
@abc.abstractmethod
def key_sizes(self) -> typing.FrozenSet[int]:
    """
        Valid key sizes for this algorithm in bits
        """