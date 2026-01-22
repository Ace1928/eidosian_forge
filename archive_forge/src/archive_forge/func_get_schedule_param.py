from functools import wraps
import weakref
import abc
import warnings
from ..data_sparsifier import BaseDataSparsifier
@abc.abstractmethod
def get_schedule_param(self):
    """
        Abstract method that needs to be implemented by the child class.
        The expected return type should is a dictionary of name to schedule_param value
        The returned values will be updated in sparsifier when the scheduler step() function
        is called.

        Example:
            >>> def get_schedule_param(self):
            ...     new_param = {}
            ...     for name in self.sparsifier.data_groups.keys():
            ...         new_param[name] = self.sparsifier.data_groups[name][self.schedule_param] * 0.5
            ...     return new_param

        When the step() function is called, the value in self.sparsifier.data_groups[name][self.schedule_param]
        would be halved
        """
    raise NotImplementedError