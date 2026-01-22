from __future__ import annotations
import hashlib
from typing import TYPE_CHECKING, Union, cast
from typing_extensions import TypeAlias
from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.GraphVizChart_pb2 import GraphVizChart as GraphVizChartProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.util import HASHLIB_KWARGS
@gather_metrics('graphviz_chart')
def graphviz_chart(self, figure_or_dot: FigureOrDot, use_container_width: bool=False) -> DeltaGenerator:
    """Display a graph using the dagre-d3 library.

        Parameters
        ----------
        figure_or_dot : graphviz.dot.Graph, graphviz.dot.Digraph, str
            The Graphlib graph object or dot string to display

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over the figure's native `width` value.

        Example
        -------
        >>> import streamlit as st
        >>> import graphviz
        >>>
        >>> # Create a graphlib graph object
        >>> graph = graphviz.Digraph()
        >>> graph.edge('run', 'intr')
        >>> graph.edge('intr', 'runbl')
        >>> graph.edge('runbl', 'run')
        >>> graph.edge('run', 'kernel')
        >>> graph.edge('kernel', 'zombie')
        >>> graph.edge('kernel', 'sleep')
        >>> graph.edge('kernel', 'runmem')
        >>> graph.edge('sleep', 'swap')
        >>> graph.edge('swap', 'runswap')
        >>> graph.edge('runswap', 'new')
        >>> graph.edge('runswap', 'runmem')
        >>> graph.edge('new', 'runmem')
        >>> graph.edge('sleep', 'runmem')
        >>>
        >>> st.graphviz_chart(graph)

        Or you can render the chart from the graph using GraphViz's Dot
        language:

        >>> st.graphviz_chart('''
            digraph {
                run -> intr
                intr -> runbl
                runbl -> run
                run -> kernel
                kernel -> zombie
                kernel -> sleep
                kernel -> runmem
                sleep -> swap
                swap -> runswap
                runswap -> new
                runswap -> runmem
                new -> runmem
                sleep -> runmem
            }
        ''')

        .. output::
           https://doc-graphviz-chart.streamlit.app/
           height: 600px

        """
    delta_path = self.dg._get_delta_path_str()
    element_id = hashlib.md5(delta_path.encode(), **HASHLIB_KWARGS).hexdigest()
    graphviz_chart_proto = GraphVizChartProto()
    marshall(graphviz_chart_proto, figure_or_dot, use_container_width, element_id)
    return self.dg._enqueue('graphviz_chart', graphviz_chart_proto)