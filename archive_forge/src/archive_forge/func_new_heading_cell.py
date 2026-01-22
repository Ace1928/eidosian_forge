from __future__ import annotations
import warnings
from nbformat._struct import Struct
def new_heading_cell(source=None, level=1, rendered=None, metadata=None):
    """Create a new section cell with a given integer level."""
    cell = NotebookNode()
    cell.cell_type = 'heading'
    if source is not None:
        cell.source = str_passthrough(source)
    cell.level = int(level)
    cell.metadata = NotebookNode(metadata or {})
    return cell