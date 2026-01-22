from __future__ import division, print_function, absolute_import
import locale
from petl.compat import izip_longest, next, xrange, BytesIO
from petl.util.base import Table
from petl.io.sources import read_source_from_arg, write_source_from_arg
def toxls(tbl, filename, sheet, encoding=None, style_compression=0, styles=None):
    """
    Write a table to a new Excel .xls file.

    """
    import xlwt
    if encoding is None:
        encoding = locale.getpreferredencoding()
    wb = xlwt.Workbook(encoding=encoding, style_compression=style_compression)
    ws = wb.add_sheet(sheet)
    if styles is None:
        for r, row in enumerate(tbl):
            for c, v in enumerate(row):
                ws.write(r, c, label=v)
    else:
        it = iter(tbl)
        try:
            hdr = next(it)
            flds = list(map(str, hdr))
            for c, f in enumerate(flds):
                ws.write(0, c, label=f)
                if f not in styles or styles[f] is None:
                    styles[f] = xlwt.Style.default_style
        except StopIteration:
            pass
        styles = [styles[f] for f in flds]
        for r, row in enumerate(it):
            for c, (v, style) in enumerate(izip_longest(row, styles, fillvalue=None)):
                ws.write(r + 1, c, label=v, style=style)
    target = write_source_from_arg(filename)
    with target.open('wb') as target2:
        wb.save(target2)