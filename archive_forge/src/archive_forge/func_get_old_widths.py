import io
import math
import os
import typing
import weakref
def get_old_widths(xref):
    """Retrieve old font '/W' and '/DW' values."""
    df = doc.xref_get_key(xref, 'DescendantFonts')
    if df[0] != 'array':
        return (None, None)
    df_xref = int(df[1][1:-1].replace('0 R', ''))
    widths = doc.xref_get_key(df_xref, 'W')
    if widths[0] != 'array':
        widths = None
    else:
        widths = widths[1]
    dwidths = doc.xref_get_key(df_xref, 'DW')
    if dwidths[0] != 'int':
        dwidths = None
    else:
        dwidths = dwidths[1]
    return (widths, dwidths)