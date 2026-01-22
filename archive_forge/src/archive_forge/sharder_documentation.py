import abc
import torch.nn as nn

        Shard a module base on the implementation of this method, and
        return the sharded version of the module.

        Args:
            module (:class:`torch.nn.Module`):
                The module to apply sharding to.
        Returns:
            A :class:`torch.nn.Module` object that represents a module
            that's already been sharded.
        