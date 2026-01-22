from .. import functions as fn
from ..Qt import QtWidgets
from .GraphicsWidget import GraphicsWidget
from .LabelItem import LabelItem
from .PlotItem import PlotItem
from .ViewBox import ViewBox
def getItem(self, row, col):
    """Return the item in (*row*, *col*). If the cell is empty, return None."""
    return self.rows.get(row, {}).get(col, None)