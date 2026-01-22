import io
import math
import os
import typing
import weakref
def do_links(doc1: fitz.Document, doc2: fitz.Document, from_page: int=-1, to_page: int=-1, start_at: int=-1) -> None:
    """Insert links contained in copied page range into destination PDF.

    Parameter values **must** equal those of method insert_pdf(), which must
    have been previously executed.
    """

    def cre_annot(lnk, xref_dst, pno_src, ctm):
        """Create annotation object string for a passed-in link."""
        r = lnk['from'] * ctm
        rect = '%g %g %g %g' % tuple(r)
        if lnk['kind'] == fitz.LINK_GOTO:
            txt = fitz.annot_skel['goto1']
            idx = pno_src.index(lnk['page'])
            p = lnk['to'] * ctm
            annot = txt % (xref_dst[idx], p.x, p.y, lnk['zoom'], rect)
        elif lnk['kind'] == fitz.LINK_GOTOR:
            if lnk['page'] >= 0:
                txt = fitz.annot_skel['gotor1']
                pnt = lnk.get('to', fitz.Point(0, 0))
                if type(pnt) is not fitz.Point:
                    pnt = fitz.Point(0, 0)
                annot = txt % (lnk['page'], pnt.x, pnt.y, lnk['zoom'], lnk['file'], lnk['file'], rect)
            else:
                txt = fitz.annot_skel['gotor2']
                to = fitz.get_pdf_str(lnk['to'])
                to = to[1:-1]
                f = lnk['file']
                annot = txt % (to, f, rect)
        elif lnk['kind'] == fitz.LINK_LAUNCH:
            txt = fitz.annot_skel['launch']
            annot = txt % (lnk['file'], lnk['file'], rect)
        elif lnk['kind'] == fitz.LINK_URI:
            txt = fitz.annot_skel['uri']
            annot = txt % (lnk['uri'], rect)
        else:
            annot = ''
        return annot
    if from_page < 0:
        fp = 0
    elif from_page >= doc2.page_count:
        fp = doc2.page_count - 1
    else:
        fp = from_page
    if to_page < 0 or to_page >= doc2.page_count:
        tp = doc2.page_count - 1
    else:
        tp = to_page
    if start_at < 0:
        raise ValueError("'start_at' must be >= 0")
    sa = start_at
    incr = 1 if fp <= tp else -1
    pno_src = list(range(fp, tp + incr, incr))
    pno_dst = [sa + i for i in range(len(pno_src))]
    xref_src = []
    xref_dst = []
    for i in range(len(pno_src)):
        p_src = pno_src[i]
        p_dst = pno_dst[i]
        old_xref = doc2.page_xref(p_src)
        new_xref = doc1.page_xref(p_dst)
        xref_src.append(old_xref)
        xref_dst.append(new_xref)
    for i in range(len(xref_src)):
        page_src = doc2[pno_src[i]]
        links = page_src.get_links()
        if len(links) == 0:
            page_src = None
            continue
        ctm = ~page_src.transformation_matrix
        page_dst = doc1[pno_dst[i]]
        link_tab = []
        for l in links:
            if l['kind'] == fitz.LINK_GOTO and l['page'] not in pno_src:
                continue
            annot_text = cre_annot(l, xref_dst, pno_src, ctm)
            if annot_text:
                link_tab.append(annot_text)
        if link_tab != []:
            page_dst._addAnnot_FromString(tuple(link_tab))