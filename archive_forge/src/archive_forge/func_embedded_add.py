import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def embedded_add(args):
    """Insert a new embedded file."""
    doc = open_file(args.input, args.password, pdf=True)
    if not doc.can_save_incrementally() and (args.output is None or args.output == args.input):
        sys.exit('cannot save PDF incrementally')
    try:
        doc.embfile_del(args.name)
        sys.exit("entry '%s' already exists" % args.name)
    except Exception:
        pass
    if not os.path.exists(args.path) or not os.path.isfile(args.path):
        sys.exit("no such file '%s'" % args.path)
    with open(args.path, 'rb') as f:
        stream = f.read()
    filename = args.path
    ufilename = filename
    if not args.desc:
        desc = filename
    else:
        desc = args.desc
    doc.embfile_add(args.name, stream, filename=filename, ufilename=ufilename, desc=desc)
    if not args.output or args.output == args.input:
        doc.saveIncr()
    else:
        doc.save(args.output, garbage=3)
    doc.close()