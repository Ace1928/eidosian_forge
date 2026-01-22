from typing import (
from .constants import (
from .exceptions import (
from .utils import (
def remove_settable(self, name: str) -> None:
    """
        Convenience method for removing a settable parameter from the CommandSet

        :param name: name of the settable being removed
        :raises: KeyError if the Settable matches this name
        """
    try:
        del self._settables[name]
    except KeyError:
        raise KeyError(name + ' is not a settable parameter')