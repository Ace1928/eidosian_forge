import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def visit_entry(self, node):
    self.column_count += 1
    cellspec_name = self.rststyle('%s%%d.%%c%%d' % TABLESTYLEPREFIX, (self.table_count, 'A', 1))
    attrib = {'table:style-name': cellspec_name, 'office:value-type': 'string'}
    morecols = node.get('morecols', 0)
    if morecols > 0:
        attrib['table:number-columns-spanned'] = '%d' % (morecols + 1,)
        self.column_count += morecols
    morerows = node.get('morerows', 0)
    if morerows > 0:
        attrib['table:number-rows-spanned'] = '%d' % (morerows + 1,)
    el1 = self.append_child('table:table-cell', attrib=attrib)
    self.set_current_element(el1)