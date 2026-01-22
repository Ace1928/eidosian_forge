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
def visit_topic(self, node):
    if 'classes' in node.attributes:
        if 'contents' in node.attributes['classes']:
            label = self.find_title_label(node, docutils.nodes.title, 'contents')
            if self.settings.generate_oowriter_toc:
                el1 = self.append_child('text:table-of-content', attrib={'text:name': 'Table of Contents1', 'text:protected': 'true', 'text:style-name': 'Sect1'})
                el2 = SubElement(el1, 'text:table-of-content-source', attrib={'text:outline-level': '10'})
                el3 = SubElement(el2, 'text:index-title-template', attrib={'text:style-name': 'Contents_20_Heading'})
                el3.text = label
                self.generate_table_of_content_entry_template(el2)
                el4 = SubElement(el1, 'text:index-body')
                el5 = SubElement(el4, 'text:index-title')
                el6 = SubElement(el5, 'text:p', attrib={'text:style-name': self.rststyle('contents-heading')})
                el6.text = label
                self.save_current_element = self.current_element
                self.table_of_content_index_body = el4
                self.set_current_element(el4)
            else:
                el = self.append_p('horizontalline')
                el = self.append_p('centeredtextbody')
                el1 = SubElement(el, 'text:span', attrib={'text:style-name': self.rststyle('strong')})
                el1.text = label
            self.in_table_of_contents = True
        elif 'abstract' in node.attributes['classes']:
            el = self.append_p('horizontalline')
            el = self.append_p('centeredtextbody')
            el1 = SubElement(el, 'text:span', attrib={'text:style-name': self.rststyle('strong')})
            label = self.find_title_label(node, docutils.nodes.title, 'abstract')
            el1.text = label
        elif 'dedication' in node.attributes['classes']:
            el = self.append_p('horizontalline')
            el = self.append_p('centeredtextbody')
            el1 = SubElement(el, 'text:span', attrib={'text:style-name': self.rststyle('strong')})
            label = self.find_title_label(node, docutils.nodes.title, 'dedication')
            el1.text = label