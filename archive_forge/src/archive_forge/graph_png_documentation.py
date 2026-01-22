from typing import Any, Optional
from langchain_core.runnables.graph import Graph, LabelsDict

        Draws the given state graph into a PNG file.
        Requires graphviz and pygraphviz to be installed.
        :param graph: The graph to draw
        :param output_path: The path to save the PNG. If None, PNG bytes are returned.
        