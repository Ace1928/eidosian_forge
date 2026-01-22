import math, sys, os, codecs, base64
from io import BytesIO, StringIO
from reportlab.pdfbase.pdfmetrics import stringWidth # for font info
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative
from reportlab.graphics.renderbase import getStateDelta, Renderer, renderScaledDrawing
from reportlab.graphics.shapes import STATE_DEFAULTS, Path, UserNode
from reportlab.graphics.shapes import * # (only for test0)
from reportlab import rl_config
from reportlab.lib.utils import RLString, isUnicode, isBytes
from reportlab.pdfgen.canvas import FILL_EVEN_ODD, FILL_NON_ZERO
from .renderPM import _getImage
from xml.dom import getDOMImplementation
def transformNode(doc, newTag, node=None, **attrDict):
    """Transform a DOM node into new node and copy selected attributes.

    Creates a new DOM node with tag name 'newTag' for document 'doc'
    and copies selected attributes from an existing 'node' as provided
    in 'attrDict'. The source 'node' can be None. Attribute values will
    be converted to strings.

    E.g.

        n = transformNode(doc, "node1", x="0", y="1")
        -> DOM node for <node1 x="0" y="1"/>

        n = transformNode(doc, "node1", x=0, y=1+1)
        -> DOM node for <node1 x="0" y="2"/>

        n = transformNode(doc, "node1", node0, x="x0", y="x0", zoo=bar())
        -> DOM node for <node1 x="[node0.x0]" y="[node0.y0]" zoo="[bar()]"/>
    """
    newNode = doc.createElement(newTag)
    for newAttr, attr in attrDict.items():
        sattr = str(attr)
        if not node:
            newNode.setAttribute(newAttr, sattr)
        else:
            attrVal = node.getAttribute(sattr)
            newNode.setAttribute(newAttr, attrVal or sattr)
    return newNode