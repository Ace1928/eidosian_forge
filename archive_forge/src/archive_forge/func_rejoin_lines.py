from __future__ import annotations
from base64 import decodebytes, encodebytes
def rejoin_lines(nb):
    """rejoin multiline text into strings

    For reversing effects of ``split_lines(nb)``.

    This only rejoins lines that have been split, so if text objects were not split
    they will pass through unchanged.

    Used when reading JSON files that may have been passed through split_lines.
    """
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type == 'code':
                if 'input' in cell and isinstance(cell.input, list):
                    cell.input = '\n'.join(cell.input)
                for output in cell.outputs:
                    for key in _multiline_outputs:
                        item = output.get(key, None)
                        if isinstance(item, list):
                            output[key] = '\n'.join(item)
            else:
                for key in ['source', 'rendered']:
                    item = cell.get(key, None)
                    if isinstance(item, list):
                        cell[key] = '\n'.join(item)
    return nb