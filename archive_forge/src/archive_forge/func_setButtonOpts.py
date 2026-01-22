import warnings
from ...Qt import QtCore
from .action import ParameterControlledButton
from .basetypes import GroupParameter, GroupParameterItem
from ..ParameterItem import ParameterItem
from ...Qt import QtCore, QtWidgets
def setButtonOpts(self, **opts):
    """
        Update individual button options without replacing the entire
        button definition.
        """
    buttonOpts = self.opts.get('button', {}).copy()
    buttonOpts.update(opts)
    self.setOpts(button=buttonOpts)