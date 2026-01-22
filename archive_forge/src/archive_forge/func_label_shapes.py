import logging
import copy
from ..initializer import Uniform
from .base_module import BaseModule
@property
def label_shapes(self):
    """Gets label shapes.

        Returns
        -------
        list
            A list of `(name, shape)` pairs. The return value could be `None` if
            the module does not need labels, or if the module is not bound for
            training (in this case, label information is not available).
        """
    assert self.binded
    return self._label_shapes