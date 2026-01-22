import logging
import os.path
import re
import sys
from typing import Any, Container, Dict, Iterable, List, Optional, TextIO, Union, cast
from argparse import ArgumentParser
import pdfminer
from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines, PDFXRefFallback
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import PDFObjectNotFound, PDFValueError
from pdfminer.pdftypes import PDFStream, PDFObjRef, resolve1, stream_value
from pdfminer.psparser import PSKeyword, PSLiteral, LIT
from pdfminer.utils import isnumber
def dumppdf(outfp: TextIO, fname: str, objids: Iterable[int], pagenos: Container[int], password: str='', dumpall: bool=False, codec: Optional[str]=None, extractdir: Optional[str]=None, show_fallback_xref: bool=False) -> None:
    fp = open(fname, 'rb')
    parser = PDFParser(fp)
    doc = PDFDocument(parser, password)
    if objids:
        for objid in objids:
            obj = doc.getobj(objid)
            dumpxml(outfp, obj, codec=codec)
    if pagenos:
        for pageno, page in enumerate(PDFPage.create_pages(doc)):
            if pageno in pagenos:
                if codec:
                    for obj in page.contents:
                        obj = stream_value(obj)
                        dumpxml(outfp, obj, codec=codec)
                else:
                    dumpxml(outfp, page.attrs)
    if dumpall:
        dumpallobjs(outfp, doc, codec, show_fallback_xref)
    if not objids and (not pagenos) and (not dumpall):
        dumptrailers(outfp, doc, show_fallback_xref)
    fp.close()
    if codec not in ('raw', 'binary'):
        outfp.write('\n')
    return