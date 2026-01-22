from nbconvert.preprocessors.clearoutput import ClearOutputPreprocessor
from nbclient.exceptions import CellExecutionError
from nbclient.client import NotebookClient
from traitlets import Bool, Unicode
def strip_notebook_errors(self, nb):
    """Strip error messages and traceback from a Notebook."""
    cells = nb['cells']
    code_cells = [cell for cell in cells if cell['cell_type'] == 'code']
    for cell in code_cells:
        strip_code_cell_warnings(cell)
        self.strip_code_cell_errors(cell)
    return nb