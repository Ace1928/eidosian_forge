import warnings
import numpy as np
from .activations import Activation
from ..block import Block, HybridBlock
from ..utils import _indent
from ... import nd, sym
from ...util import is_np_array
def hybridize(self, active=True, **kwargs):
    """Activates or deactivates `HybridBlock` s recursively. Has no effect on
        non-hybrid children.

        Parameters
        ----------
        active : bool, default True
            Whether to turn hybrid on or off.
        **kwargs : string
            Additional flags for hybridized operator.
        """
    if self._children and all((isinstance(c, HybridBlock) for c in self._children.values())):
        warnings.warn("All children of this Sequential layer '%s' are HybridBlocks. Consider using HybridSequential for the best performance." % self.prefix, stacklevel=2)
    super(Sequential, self).hybridize(active, **kwargs)