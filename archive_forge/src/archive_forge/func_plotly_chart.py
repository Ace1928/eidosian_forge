from __future__ import annotations
import json
import urllib.parse
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Union, cast
from typing_extensions import TypeAlias
from streamlit import type_util
from streamlit.elements.lib.streamlit_plotly_theme import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.PlotlyChart_pb2 import PlotlyChart as PlotlyChartProto
from streamlit.runtime.legacy_caching import caching
from streamlit.runtime.metrics_util import gather_metrics
@gather_metrics('plotly_chart')
def plotly_chart(self, figure_or_data: FigureOrData, use_container_width: bool=False, sharing: SharingMode='streamlit', theme: Literal['streamlit'] | None='streamlit', **kwargs: Any) -> DeltaGenerator:
    """Display an interactive Plotly chart.

        Plotly is a charting library for Python. The arguments to this function
        closely follow the ones for Plotly's `plot()` function. You can find
        more about Plotly at https://plot.ly/python.

        To show Plotly charts in Streamlit, call `st.plotly_chart` wherever you
        would call Plotly's `py.plot` or `py.iplot`.

        Parameters
        ----------
        figure_or_data : plotly.graph_objs.Figure, plotly.graph_objs.Data,            dict/list of plotly.graph_objs.Figure/Data

            See https://plot.ly/python/ for examples of graph descriptions.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over the figure's native `width` value.

        sharing : "streamlit", "private", "secret", or "public"
            Use "streamlit" to insert the plot and all its dependencies
            directly in the Streamlit app using plotly's offline mode (default).
            Use any other sharing mode to send the chart to Plotly chart studio, which
            requires an account. See https://plot.ly/python/chart-studio/ for more information.

        theme : "streamlit" or None
            The theme of the chart. Currently, we only support "streamlit" for the Streamlit
            defined design or None to fallback to the default behavior of the library.

        **kwargs
            Any argument accepted by Plotly's `plot()` function.

        Example
        -------
        The example below comes straight from the examples at
        https://plot.ly/python:

        >>> import streamlit as st
        >>> import numpy as np
        >>> import plotly.figure_factory as ff
        >>>
        >>> # Add histogram data
        >>> x1 = np.random.randn(200) - 2
        >>> x2 = np.random.randn(200)
        >>> x3 = np.random.randn(200) + 2
        >>>
        >>> # Group data together
        >>> hist_data = [x1, x2, x3]
        >>>
        >>> group_labels = ['Group 1', 'Group 2', 'Group 3']
        >>>
        >>> # Create distplot with custom bin_size
        >>> fig = ff.create_distplot(
        ...         hist_data, group_labels, bin_size=[.1, .25, .5])
        >>>
        >>> # Plot!
        >>> st.plotly_chart(fig, use_container_width=True)

        .. output::
           https://doc-plotly-chart.streamlit.app/
           height: 400px

        """
    plotly_chart_proto = PlotlyChartProto()
    if theme != 'streamlit' and theme != None:
        raise StreamlitAPIException(f'You set theme="{theme}" while Streamlit charts only support theme=”streamlit” or theme=None to fallback to the default library theme.')
    marshall(plotly_chart_proto, figure_or_data, use_container_width, sharing, theme, **kwargs)
    return self.dg._enqueue('plotly_chart', plotly_chart_proto)