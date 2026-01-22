from reportlab.lib import colors
from ._Graph import GraphData
def new_graph(self, data, name=None, style='bar', color=colors.lightgreen, altcolor=colors.darkseagreen, linewidth=1, center=None, colour=None, altcolour=None, centre=None):
    """Add a GraphData object to the diagram.

        Arguments:
         - data      List of (position, value) int tuples
         - name      String, description of the graph
         - style     String ('bar', 'heat', 'line') describing how the graph
           will be drawn
         - color    colors.Color describing the color to draw all or 'high'
           (some styles) data (overridden by backwards compatible
           argument with UK spelling, colour).
         - altcolor  colors.Color describing the color to draw 'low' (some
           styles) data (overridden by backwards compatible argument
           with UK spelling, colour).
         - linewidth     Float describing linewidth for graph
         - center        Float setting the value at which the x-axis
           crosses the y-axis (overridden by backwards
           compatible argument with UK spelling, centre)

        Add a GraphData object to the diagram (will be stored internally).
        """
    if colour is not None:
        color = colour
    if altcolour is not None:
        altcolor = altcolour
    if centre is not None:
        center = centre
    id = self._next_id
    graph = GraphData(id, data, name, style, color, altcolor, center)
    graph.linewidth = linewidth
    self._graphs[id] = graph
    self._next_id += 1
    return graph