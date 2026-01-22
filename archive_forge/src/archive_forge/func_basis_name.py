import os
import abc
import numbers
import numpy as np
from . import polyutils as pu
@property
@abc.abstractmethod
def basis_name(self):
    pass