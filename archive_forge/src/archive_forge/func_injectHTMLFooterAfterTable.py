import base64
import json
import logging
import re
import uuid
from xml.dom import minidom
from IPython.display import HTML, display
from rdkit import Chem
from rdkit.Chem import Draw
from . import rdMolDraw2D
def injectHTMLFooterAfterTable(html):
    doc = minidom.parseString(html.replace(' scoped', ''))
    tables = doc.getElementsByTagName('table')
    for table in tables:
        tbody = table.getElementsByTagName('tbody')
        if tbody:
            if len(tbody) != 1:
                return html
            tbody = tbody.pop(0)
        else:
            tbody = table
        div_list = tbody.getElementsByTagName('div')
        if any((div.getAttribute('class') == 'rdk-str-rnr-mol-container' for div in div_list)):
            return generateHTMLFooter(doc, table)
    return html