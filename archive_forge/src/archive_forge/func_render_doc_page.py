from __future__ import print_function
import argparse
import os
import sys
from importlib import import_module
from jinja2 import Template
from palettable.palette import Palette
def render_doc_page(dir_, palette_names, palette_dict):
    """
    Render the documentation page in a given directory.

    """
    print('Rendering index in dir {}'.format(dir_))
    with open(os.path.join(dir_, 'index.md.tpl')) as f:
        tpl = Template(f.read())
    with open(os.path.join(dir_, 'index.md'), 'w') as f:
        f.write(tpl.render(palettes=palette_names, palette_dict=palette_dict))