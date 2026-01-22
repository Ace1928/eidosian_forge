import io
import math
import os
import typing
import weakref
def getLinkDict(ln, document=None) -> dict:
    if isinstance(ln, fitz.Outline):
        dest = ln.destination(document)
    elif isinstance(ln, fitz.Link):
        dest = ln.dest
    else:
        assert 0, f'Unexpected type(ln)={type(ln)!r}.'
    nl = {'kind': dest.kind, 'xref': 0}
    try:
        nl['from'] = ln.rect
    except Exception:
        if g_exceptions_verbose:
            fitz.exception_info()
        pass
    pnt = fitz.Point(0, 0)
    if dest.flags & fitz.LINK_FLAG_L_VALID:
        pnt.x = dest.lt.x
    if dest.flags & fitz.LINK_FLAG_T_VALID:
        pnt.y = dest.lt.y
    if dest.kind == fitz.LINK_URI:
        nl['uri'] = dest.uri
    elif dest.kind == fitz.LINK_GOTO:
        nl['page'] = dest.page
        nl['to'] = pnt
        if dest.flags & fitz.LINK_FLAG_R_IS_ZOOM:
            nl['zoom'] = dest.rb.x
        else:
            nl['zoom'] = 0.0
    elif dest.kind == fitz.LINK_GOTOR:
        nl['file'] = dest.file_spec.replace('\\', '/')
        nl['page'] = dest.page
        if dest.page < 0:
            nl['to'] = dest.dest
        else:
            nl['to'] = pnt
            if dest.flags & fitz.LINK_FLAG_R_IS_ZOOM:
                nl['zoom'] = dest.rb.x
            else:
                nl['zoom'] = 0.0
    elif dest.kind == fitz.LINK_LAUNCH:
        nl['file'] = dest.file_spec.replace('\\', '/')
    elif dest.kind == fitz.LINK_NAMED:
        assert not dest.named.keys() & nl.keys()
        nl.update(dest.named)
        if 'to' in nl:
            nl['to'] = fitz.Point(nl['to'])
    else:
        nl['page'] = dest.page
    return nl