from plotly import exceptions, optional_imports
from plotly.graph_objs import graph_objs
def get_table_matrix(self):
    """
        Create z matrix to make heatmap with striped table coloring

        :rtype (list[list]) table_matrix: z matrix to make heatmap with striped
            table coloring.
        """
    header = [0] * len(self.table_text[0])
    odd_row = [0.5] * len(self.table_text[0])
    even_row = [1] * len(self.table_text[0])
    table_matrix = [None] * len(self.table_text)
    table_matrix[0] = header
    for i in range(1, len(self.table_text), 2):
        table_matrix[i] = odd_row
    for i in range(2, len(self.table_text), 2):
        table_matrix[i] = even_row
    if self.index:
        for array in table_matrix:
            array[0] = 0
    return table_matrix