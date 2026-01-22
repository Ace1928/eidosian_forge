from __future__ import annotations
from typing import Any, List, NamedTuple, Optional, Tuple
def draw_graphviz(self, **kwargs: Any) -> None:
    """
        Provides better drawing

        Usage in a jupyter notebook:

            >>> from IPython.display import SVG
            >>> self.draw_graphviz_svg(layout="dot", filename="web.svg")
            >>> SVG('web.svg')
        """
    from networkx.drawing.nx_agraph import to_agraph
    try:
        import pygraphviz
    except ImportError as e:
        if e.name == '_graphviz':
            '\n                >>> e.msg  # pygraphviz throws this error\n                ImportError: libcgraph.so.6: cannot open shared object file\n                '
            raise ImportError('Could not import graphviz debian package. Please install it with:`sudo apt-get update``sudo apt-get install graphviz graphviz-dev`')
        else:
            raise ImportError('Could not import pygraphviz python package. Please install it with:`pip install pygraphviz`.')
    graph = to_agraph(self._graph)
    graph.layout(prog=kwargs.get('prog', 'dot'))
    graph.draw(kwargs.get('path', 'graph.svg'))