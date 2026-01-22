import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def extract_objects(args):
    """Extract images and / or fonts from a PDF."""
    if not args.fonts and (not args.images):
        sys.exit('neither fonts nor images requested')
    doc = open_file(args.input, args.password, pdf=True)
    if args.pages:
        pages = get_list(args.pages, doc.page_count + 1)
    else:
        pages = range(1, doc.page_count + 1)
    if not args.output:
        out_dir = os.path.abspath(os.curdir)
    else:
        out_dir = args.output
        if not (os.path.exists(out_dir) and os.path.isdir(out_dir)):
            sys.exit('output directory %s does not exist' % out_dir)
    font_xrefs = set()
    image_xrefs = set()
    for pno in pages:
        if args.fonts:
            itemlist = doc.get_page_fonts(pno - 1)
            for item in itemlist:
                xref = item[0]
                if xref not in font_xrefs:
                    font_xrefs.add(xref)
                    fontname, ext, _, buffer = doc.extract_font(xref)
                    if ext == 'n/a' or not buffer:
                        continue
                    outname = os.path.join(out_dir, f'{fontname.replace(' ', '-')}-{xref}.{ext}')
                    with open(outname, 'wb') as outfile:
                        outfile.write(buffer)
                    buffer = None
        if args.images:
            itemlist = doc.get_page_images(pno - 1)
            for item in itemlist:
                xref = item[0]
                if xref not in image_xrefs:
                    image_xrefs.add(xref)
                    pix = recoverpix(doc, item)
                    if type(pix) is dict:
                        ext = pix['ext']
                        imgdata = pix['image']
                        outname = os.path.join(out_dir, 'img-%i.%s' % (xref, ext))
                        with open(outname, 'wb') as outfile:
                            outfile.write(imgdata)
                    else:
                        outname = os.path.join(out_dir, 'img-%i.png' % xref)
                        pix2 = pix if pix.colorspace.n < 4 else fitz.Pixmap(fitz.csRGB, pix)
                        pix2.save(outname)
    if args.fonts:
        fitz.message("saved %i fonts to '%s'" % (len(font_xrefs), out_dir))
    if args.images:
        fitz.message("saved %i images to '%s'" % (len(image_xrefs), out_dir))
    doc.close()