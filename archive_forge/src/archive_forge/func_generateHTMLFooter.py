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
def generateHTMLFooter(doc, element):
    element_parent = element.parentNode
    if element_parent.nodeName.lower() != 'div':
        element_parent = doc.createElement('div')
        element_parent.appendChild(element)
    script = doc.createElement('script')
    script.setAttribute('type', 'module')
    cmd = doc.createTextNode(f'if (window.rdkStrRnr) {{\n  window.rdkStrRnr.then(\n    function(Renderer) {{\n      if (Renderer) {{\n        Renderer.updateMolDrawDivs();\n      }}\n    }}\n  ).catch();\n}}')
    script.appendChild(cmd)
    element_parent.appendChild(script)
    html = element_parent.toxml()
    return html