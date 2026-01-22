import sys
import tkinter
from tkinter import ttk
from .gui_utilities import UniformDictController, ScrollableFrame
from .geodesics import geodesic_index_to_color, LengthSpectrumError
from ..drilling.exceptions import WordAppearsToBeParabolic
from ..SnapPy import word_as_list # type: ignore
def update_geodesic_data(self):
    success = self.raytracing_view.update_geodesic_data_and_redraw()
    if success:
        self.status_label.configure(text=_default_status_msg, foreground='')
    else:
        self.status_label.configure(text='Limiting size of geodesic tube to prevent intersection with core curve.', foreground='red')