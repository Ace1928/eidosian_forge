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
def get_table_style(self, node):
    table_style = None
    table_name = None
    str_classes = node.get('classes')
    if str_classes is not None:
        for str_class in str_classes:
            if str_class.startswith(TABLESTYLEPREFIX):
                table_name = str_class
                break
    if table_name is not None:
        table_style = self.table_styles.get(table_name)
        if table_style is None:
            self.document.reporter.warning('Can\'t find table style "%s".  Using default.' % (table_name,))
            table_name = TABLENAMEDEFAULT
            table_style = self.table_styles.get(table_name)
            if table_style is None:
                self.document.reporter.warning('Can\'t find default table style "%s".  Using built-in default.' % (table_name,))
                table_style = BUILTIN_DEFAULT_TABLE_STYLE
    else:
        table_name = TABLENAMEDEFAULT
        table_style = self.table_styles.get(table_name)
        if table_style is None:
            self.document.reporter.warning('Can\'t find default table style "%s".  Using built-in default.' % (table_name,))
            table_style = BUILTIN_DEFAULT_TABLE_STYLE
    return table_style