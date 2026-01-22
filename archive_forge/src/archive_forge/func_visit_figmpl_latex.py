from docutils import nodes
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.images import Figure, Image
import os
from os.path import relpath
from pathlib import PurePath, Path
import shutil
from sphinx.errors import ExtensionError
import matplotlib
def visit_figmpl_latex(self, node):
    if node['srcset'] is not None:
        imagedir, srcset = _copy_images_figmpl(self, node)
        maxmult = -1
        maxmult = max(srcset, default=-1)
        node['uri'] = PurePath(srcset[maxmult]).name
    self.visit_figure(node)