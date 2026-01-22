import importlib
from .. import ItemGroup, SRTTransform, debug
from .. import functions as fn
from ..graphicsItems.ROI import ROI
from ..Qt import QT_LIB, QtCore, QtWidgets
from . import TransformGuiTemplate_generic as ui_template
def selectionChanged(self, sel, multi):
    """
        Inform the item that its selection state has changed. 
        ============== =========================================================
        **Arguments:**
        sel            (bool) whether the item is currently selected
        multi          (bool) whether there are multiple items currently 
                       selected
        ============== =========================================================
        """
    self.selectedAlone = sel and (not multi)
    self.showSelectBox()
    if self.selectedAlone:
        self.ctrlWidget().show()
    else:
        self.ctrlWidget().hide()