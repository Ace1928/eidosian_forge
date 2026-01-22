from ...widgets.SpinBox import SpinBox
from .basetypes import WidgetParameterItem
def showEditor(self):
    super().showEditor()
    self.widget.selectNumber()