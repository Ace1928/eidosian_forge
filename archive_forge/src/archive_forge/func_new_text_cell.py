from __future__ import annotations
from nbformat._struct import Struct
def new_text_cell(cell_type, source=None, rendered=None):
    """Create a new text cell."""
    cell = NotebookNode()
    if source is not None:
        cell.source = str(source)
    if rendered is not None:
        cell.rendered = str(rendered)
    cell.cell_type = cell_type
    return cell