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
def create_custom_headfoot(self, parent, text, style_name, automatic_styles):
    parent = SubElement(parent, 'text:p', attrib={'text:style-name': self.rststyle(style_name)})
    current_element = None
    field_iter = self.split_field_specifiers_iter(text)
    for item in field_iter:
        if item[0] == ODFTranslator.code_field:
            if item[1] not in ('p', 'P', 't1', 't2', 't3', 't4', 'd1', 'd2', 'd3', 'd4', 'd5', 's', 't', 'a'):
                msg = 'bad field spec: %%%s%%' % (item[1],)
                raise RuntimeError(msg)
            el1 = self.make_field_element(parent, item[1], style_name, automatic_styles)
            if el1 is None:
                msg = 'bad field spec: %%%s%%' % (item[1],)
                raise RuntimeError(msg)
            else:
                current_element = el1
        elif current_element is None:
            parent.text = item[1]
        else:
            current_element.tail = item[1]