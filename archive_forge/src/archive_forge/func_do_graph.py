import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def do_graph(self):
    general_draw_string = self.graph.attr.get('_draw_', '')
    label_string = self.graph.attr.get('_ldraw_', '')
    if general_draw_string.startswith('c 5 -white C 5 -white') and (not self.graph.attr.get('style')):
        general_draw_string = ''
    if getattr(self.graph, '_draw_', None):
        general_draw_string = 'c 5 -black ' + general_draw_string
        pass
    drawstring = general_draw_string + ' ' + label_string
    if drawstring.strip():
        s = self.start_graph(self.graph)
        g = self.do_drawstring(drawstring, self.graph)
        e = self.end_graph(self.graph)
        if g.strip():
            self.body += s + g + e