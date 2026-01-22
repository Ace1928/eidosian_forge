import numpy as np
from ..Qt import QtCore, QtGui, QtWidgets
def setSortMode(self, mode):
    """
        Set the mode used to sort this item against others in its column.
        
        ============== ========================================================
        **Sort Modes**
        value          Compares item.value if available; falls back to text
                       comparison.
        text           Compares item.text()
        index          Compares by the order in which items were inserted.
        ============== ========================================================
        """
    modes = ('value', 'text', 'index', None)
    if mode not in modes:
        raise ValueError('Sort mode must be one of %s' % str(modes))
    self.sortMode = mode