from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import missing_required_lib
import traceback
import sys
import os
def control_xml_utf8(filepath, module):
    if not HAS_LXML_LIBRARY:
        module.fail_json(msg=missing_required_lib('lxml'), exception=LXML_LIBRARY_IMPORT_ERROR)
    source = filepath + '/control.xml'
    with open(source, 'rb') as source:
        parser = etree.XMLParser(encoding='iso-8859-1', strip_cdata=False)
        root = etree.parse(source, parser)
    string = etree.tostring(root, xml_declaration=True, encoding='UTF-8', pretty_print=True).decode('utf8').encode('iso-8859-1')
    with open('control_utf8.xml', 'wb') as target:
        target.write(string)