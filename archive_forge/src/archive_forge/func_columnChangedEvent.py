from ..Qt import QtCore, QtGui, QtWidgets
def columnChangedEvent(self, col):
    """Called when the text in a column has been edited (or otherwise changed).
        By default, we only use changes to column 0 to rename the parameter.
        """
    if col == 0 and self.param.opts.get('title', None) is None:
        if self.ignoreNameColumnChange:
            return
        try:
            newName = self.param.setName(self.text(col))
        except Exception:
            self.setText(0, self.param.name())
            raise
        try:
            self.ignoreNameColumnChange = True
            self.nameChanged(self, newName)
        finally:
            self.ignoreNameColumnChange = False