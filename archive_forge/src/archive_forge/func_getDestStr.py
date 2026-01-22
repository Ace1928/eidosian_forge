import io
import math
import os
import typing
import weakref
def getDestStr(xref: int, ddict: dict) -> str:
    """Calculate the PDF action string.

    Notes:
        Supports Link annotations and outline items (bookmarks).
    """
    if not ddict:
        return ''
    str_goto = '/A<</S/GoTo/D[%i 0 R/XYZ %g %g %g]>>'
    str_gotor1 = '/A<</S/GoToR/D[%s /XYZ %g %g %g]/F<</F%s/UF%s/Type/Filespec>>>>'
    str_gotor2 = '/A<</S/GoToR/D%s/F<</F%s/UF%s/Type/Filespec>>>>'
    str_launch = '/A<</S/Launch/F<</F%s/UF%s/Type/Filespec>>>>'
    str_uri = '/A<</S/URI/URI%s>>'
    if type(ddict) in (int, float):
        dest = str_goto % (xref, 0, ddict, 0)
        return dest
    d_kind = ddict.get('kind', fitz.LINK_NONE)
    if d_kind == fitz.LINK_NONE:
        return ''
    if ddict['kind'] == fitz.LINK_GOTO:
        d_zoom = ddict.get('zoom', 0)
        to = ddict.get('to', fitz.Point(0, 0))
        d_left, d_top = to
        dest = str_goto % (xref, d_left, d_top, d_zoom)
        return dest
    if ddict['kind'] == fitz.LINK_URI:
        dest = str_uri % (fitz.get_pdf_str(ddict['uri']),)
        return dest
    if ddict['kind'] == fitz.LINK_LAUNCH:
        fspec = fitz.get_pdf_str(ddict['file'])
        dest = str_launch % (fspec, fspec)
        return dest
    if ddict['kind'] == fitz.LINK_GOTOR and ddict['page'] < 0:
        fspec = fitz.get_pdf_str(ddict['file'])
        dest = str_gotor2 % (fitz.get_pdf_str(ddict['to']), fspec, fspec)
        return dest
    if ddict['kind'] == fitz.LINK_GOTOR and ddict['page'] >= 0:
        fspec = fitz.get_pdf_str(ddict['file'])
        dest = str_gotor1 % (ddict['page'], ddict['to'].x, ddict['to'].y, ddict['zoom'], fspec, fspec)
        return dest
    return ''