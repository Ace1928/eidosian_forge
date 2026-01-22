import sys
import tkinter
from tkinter import ttk
from .gui_utilities import UniformDictController, ScrollableFrame
from .geodesics import geodesic_index_to_color, LengthSpectrumError
from ..drilling.exceptions import WordAppearsToBeParabolic
from ..SnapPy import word_as_list # type: ignore
def populate_geodesics_frame(self):
    for widget in self.geodesics_frame.grid_slaves():
        widget.destroy()
    row = 0
    checkbox_column = 0
    color_column = 1
    words_column = 2
    length_column = 3
    radius_column = 5
    for geodesic in self.raytracing_view.geodesics.geodesics_sorted_by_length():
        if not geodesic.geodesic_info.core_curve_cusp:
            UniformDictController.create_checkbox(self.geodesics_frame, self.raytracing_view.ui_parameter_dict, key='geodesicTubeEnables', index=geodesic.index, row=row, column=checkbox_column, update_function=self.geodesic_checkbox_clicked)
        text = ', '.join(geodesic.words)
        if not geodesic.is_primitive():
            text += ' (not primitive)'
        l = ttk.Label(self.geodesics_frame, text=text)
        l.grid(row=row, column=words_column)
        l = ttk.Label(self.geodesics_frame, text='%.8f' % geodesic.complex_length.real())
        l.grid(row=row, column=length_column)
        im_length = geodesic.complex_length.imag()
        abs_im_length = im_length.abs()
        if abs_im_length > 1e-10:
            s = '+' if im_length > 0 else '-'
            l = ttk.Label(self.geodesics_frame, text=s + ' %.8f * I' % abs_im_length)
            l.grid(row=row, column=length_column + 1)
        color = geodesic_index_to_color(geodesic.index)
        if geodesic.geodesic_info.core_curve_cusp:
            cusp_index = geodesic.geodesic_info.core_curve_cusp.Index
            l = tkinter.Label(self.geodesics_frame, text='Cusp %d' % cusp_index)
        else:
            l = tkinter.Label(self.geodesics_frame, text='Color', fg=color_to_tkinter(color), bg=color_to_tkinter(color))
        l.grid(row=row, column=color_column, padx=5)
        if geodesic.geodesic_info.core_curve_cusp:
            l = tkinter.Label(self.geodesics_frame, text='Use Cusp areas tab')
            l.grid(row=row, column=radius_column, padx=5)
        else:
            scale = UniformDictController.create_horizontal_scale(self.geodesics_frame, self.raytracing_view.ui_parameter_dict, key='geodesicTubeRadii', index=geodesic.index, row=row, column=radius_column, left_end=0.0, right_end=1.0, update_function=self.update_geodesic_data, format_string='%.3f')
        row += 1
    self.scrollable_frame.set_widths()