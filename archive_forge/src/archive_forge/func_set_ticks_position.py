import inspect
import numpy as np
import matplotlib as mpl
from matplotlib import (
from . import art3d, proj3d
def set_ticks_position(self, position):
    """
        Set the ticks position.

        Parameters
        ----------
        position : {'lower', 'upper', 'both', 'default', 'none'}
            The position of the bolded axis lines, ticks, and tick labels.
        """
    if position in ['top', 'bottom']:
        _api.warn_deprecated('3.8', name=f'position={position!r}', obj_type='argument value', alternative="'upper' or 'lower'")
        return
    _api.check_in_list(['lower', 'upper', 'both', 'default', 'none'], position=position)
    self._tick_position = position