from __future__ import annotations
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pymatgen.util.plotting import pretty_plot
def get_plotly_figure(self, width=800, height=600, font_dict=None, term_zero=True, **kwargs):
    """Return plotly Figure object.

        Args:
            width: Width of the plot. Defaults to 800 px.
            height: Height of the plot. Defaults to 600 px.
            font_dict: define the font. Defaults to {"family": "Arial", "size": 24, "color": "#000000"}
            term_zero: If True append zero voltage point at the end
            **kwargs: passed to plotly.graph_objects.Layout
        """
    font_dict = font_dict or {'family': 'Arial', 'size': 24, 'color': '#000000'}
    hover_temp = 'Voltage : %{y:.2f} V'
    data = []
    working_ion_symbols = set()
    formula = set()
    for key, electrode in self._electrodes.items():
        x, y = self.get_plot_data(electrode, term_zero=term_zero)
        working_ion_symbols.add(electrode.working_ion.symbol)
        formula.add(electrode.framework_formula)
        plot_x, plot_y = ([x[0]], [y[0]])
        for i in range(1, len(x)):
            if x[i - 1] == x[i]:
                plot_x.append(None)
                plot_y.append(None)
            plot_x.append(x[i])
            plot_y.append(y[i])
        data.append(go.Scatter(x=plot_x, y=plot_y, name=key, hovertemplate=hover_temp))
    fig = go.Figure(data=data, layout=dict(title='Voltage vs. Capacity', width=width, height=height, font=font_dict, xaxis={'title': self._choose_best_x_label(formula=formula, work_ion_symbol=working_ion_symbols)}, yaxis={'title': 'Voltage (V)'}, **kwargs))
    fig.update_layout(template='plotly_white', title_x=0.5)
    return fig