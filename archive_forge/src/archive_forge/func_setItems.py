import sys
from collections import OrderedDict
from ..Qt import QtWidgets
@ignoreIndexChange
@blockIfUnchanged
def setItems(self, items):
    """
        *items* may be a list, a tuple, or a dict. 
        If a dict is given, then the keys are used to populate the combo box
        and the values will be used for both value() and setValue().
        """
    self.clear()
    self.addItems(items)