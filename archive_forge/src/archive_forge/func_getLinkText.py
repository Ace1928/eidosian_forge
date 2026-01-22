import io
import math
import os
import typing
import weakref
def getLinkText(page: fitz.Page, lnk: dict) -> str:
    ctm = page.transformation_matrix
    ictm = ~ctm
    r = lnk['from']
    rect = '%g %g %g %g' % tuple(r * ictm)
    annot = ''
    if lnk['kind'] == fitz.LINK_GOTO:
        if lnk['page'] >= 0:
            txt = fitz.annot_skel['goto1']
            pno = lnk['page']
            xref = page.parent.page_xref(pno)
            pnt = lnk.get('to', fitz.Point(0, 0))
            ipnt = pnt * ictm
            annot = txt % (xref, ipnt.x, ipnt.y, lnk.get('zoom', 0), rect)
        else:
            txt = fitz.annot_skel['goto2']
            annot = txt % (fitz.get_pdf_str(lnk['to']), rect)
    elif lnk['kind'] == fitz.LINK_GOTOR:
        if lnk['page'] >= 0:
            txt = fitz.annot_skel['gotor1']
            pnt = lnk.get('to', fitz.Point(0, 0))
            if type(pnt) is not fitz.Point:
                pnt = fitz.Point(0, 0)
            annot = txt % (lnk['page'], pnt.x, pnt.y, lnk.get('zoom', 0), lnk['file'], lnk['file'], rect)
        else:
            txt = fitz.annot_skel['gotor2']
            annot = txt % (fitz.get_pdf_str(lnk['to']), lnk['file'], rect)
    elif lnk['kind'] == fitz.LINK_LAUNCH:
        txt = fitz.annot_skel['launch']
        annot = txt % (lnk['file'], lnk['file'], rect)
    elif lnk['kind'] == fitz.LINK_URI:
        txt = fitz.annot_skel['uri']
        annot = txt % (lnk['uri'], rect)
    elif lnk['kind'] == fitz.LINK_NAMED:
        txt = fitz.annot_skel['named']
        lname = lnk.get('name')
        if lname is None:
            lname = lnk['nameddest']
        annot = txt % (lname, rect)
    if not annot:
        return annot
    link_names = dict([(x[0], x[2]) for x in page.annot_xrefs() if x[1] == fitz.PDF_ANNOT_LINK])
    old_name = lnk.get('id', '')
    if old_name and (lnk['xref'], old_name) in link_names.items():
        name = old_name
    else:
        i = 0
        stem = fitz.TOOLS.set_annot_stem() + '-L%i'
        while True:
            name = stem % i
            if name not in link_names.values():
                break
            i += 1
    annot = annot.replace('/Link', '/Link/NM(%s)' % name)
    return annot