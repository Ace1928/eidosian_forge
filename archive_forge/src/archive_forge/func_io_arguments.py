from __future__ import absolute_import, division, print_function
import argparse
import itertools as it
import multiprocessing as mp
import os
import sys
from collections import MutableSequence
import numpy as np
from .utils import integer_types
def io_arguments(parser, output_suffix='.txt', pickle=True, online=False):
    """
    Add input / output related arguments to an existing parser.

    Parameters
    ----------
    parser : argparse parser instance
        Existing argparse parser object.
    output_suffix : str, optional
        Suffix appended to the output files.
    pickle : bool, optional
        Add a 'pickle' sub-parser to the parser.
    online : bool, optional
        Add a 'online' sub-parser to the parser.

    """
    try:
        output = sys.stdout.buffer
    except AttributeError:
        output = sys.stdout
    parser.add_argument('-v', dest='verbose', action='count', help='increase verbosity level')
    sub_parsers = parser.add_subparsers(title='processing options')
    if pickle:
        sp = sub_parsers.add_parser('pickle', help='pickle processor')
        sp.set_defaults(func=pickle_processor)
        sp.add_argument('-o', dest='outfile', type=argparse.FileType('wb'), default=output, help='output file [default: STDOUT]')
    sp = sub_parsers.add_parser('single', help='single file processing')
    sp.set_defaults(func=process_single)
    sp.add_argument('infile', type=argparse.FileType('rb'), help='input audio file')
    sp.add_argument('-o', dest='outfile', type=argparse.FileType('wb'), default=output, help='output file [default: STDOUT]')
    sp.add_argument('-j', dest='num_threads', type=int, default=mp.cpu_count(), help='number of threads [default=%(default)s]')
    if online:
        sp.add_argument('--online', action='store_true', default=None, help='use online settings [default: offline]')
    sp = sub_parsers.add_parser('batch', help='batch file processing')
    sp.set_defaults(func=process_batch)
    sp.add_argument('files', nargs='+', help='files to be processed')
    sp.add_argument('-o', dest='output_dir', default=None, help='output directory [default=%(default)s]')
    sp.add_argument('-s', dest='output_suffix', default=output_suffix, help='suffix appended to the files (dot must be included if wanted) [default=%(default)s]')
    sp.add_argument('--ext', dest='strip_ext', action='store_false', help='keep the extension of the input file [default=strip it off before appending the output suffix]')
    sp.add_argument('-j', dest='num_workers', type=int, default=mp.cpu_count(), help='number of workers [default=%(default)s]')
    sp.add_argument('--shuffle', action='store_true', help='shuffle files before distributing them to the working threads [default=process them in sorted order]')
    sp.set_defaults(num_threads=1)
    if online:
        sp = sub_parsers.add_parser('online', help='online processing')
        sp.set_defaults(func=process_online)
        sp.add_argument('infile', nargs='?', type=argparse.FileType('rb'), default=None, help='input audio file (if no file is given, a stream operating on the system audio input is used)')
        sp.add_argument('-o', dest='outfile', type=argparse.FileType('wb'), default=output, help='output file [default: STDOUT]')
        sp.add_argument('-j', dest='num_threads', type=int, default=1, help='number of threads [default=%(default)s]')
        sp.set_defaults(online=True)
        sp.set_defaults(num_frames=1)
        sp.set_defaults(origin='stream')