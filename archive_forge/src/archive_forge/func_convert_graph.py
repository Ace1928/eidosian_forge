import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def convert_graph(dotsource, **kwargs):
    """Process dotsource and return LaTeX code

    Conversion options can be specified as keyword options. Example:
        convert_graph(data,format='tikz',crop=True)

    """
    parser = create_options_parser()
    options = parser.parse_args([])
    if kwargs.get('preproc', None):
        kwargs['texpreproc'] = kwargs['preproc']
        del kwargs['preproc']
    options.__dict__.update(kwargs)
    tex = main(True, dotsource, options)
    return tex