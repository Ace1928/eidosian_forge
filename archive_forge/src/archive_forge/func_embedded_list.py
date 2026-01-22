import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def embedded_list(args):
    """List embedded files."""
    doc = open_file(args.input, args.password, pdf=True)
    names = doc.embfile_names()
    if args.name is not None:
        if args.name not in names:
            sys.exit("no such embedded file '%s'" % args.name)
        else:
            fitz.message()
            fitz.message('printing 1 of %i embedded file%s:' % (len(names), 's' if len(names) > 1 else ''))
            fitz.message()
            print_dict(doc.embfile_info(args.name))
            fitz.message()
            return
    if not names:
        fitz.message("'%s' contains no embedded files" % doc.name)
        return
    if len(names) > 1:
        msg = "'%s' contains the following %i embedded files" % (doc.name, len(names))
    else:
        msg = "'%s' contains the following embedded file" % doc.name
    fitz.message(msg)
    fitz.message()
    for name in names:
        if not args.detail:
            fitz.message(name)
            continue
        _ = doc.embfile_info(name)
        print_dict(doc.embfile_info(name))
        fitz.message()
    doc.close()