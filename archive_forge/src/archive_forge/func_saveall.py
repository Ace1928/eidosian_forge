import tkinter
from tkinter.constants import *
def saveall(filename, canvas, items=None, margin=10, tounicode=None):
    doc = SVGdocument()
    for element in convert(doc, canvas, items, tounicode):
        doc.documentElement.appendChild(element)
    if items is None:
        bbox = canvas.bbox(ALL)
        if bbox is None:
            x1, y1, x2, y2 = (0, 0, 0, 0)
        else:
            x1, y1, x2, y2 = bbox
    else:
        x1 = None
        y1 = None
        x2 = None
        y2 = None
        for item in items:
            X1, Y1, X2, Y2 = canvas.bbox(item)
            if x1 is None:
                x1 = X1
                y1 = Y1
                x2 = X2
                y2 = Y2
            else:
                x1 = min(x1, X1)
                x2 = max(x2, X2)
                y1 = min(y1, Y1)
                y2 = max(y2, Y2)
    x1 -= margin
    y1 -= margin
    x2 += margin
    y2 += margin
    dx = x2 - x1
    dy = y2 - y1
    doc.documentElement.setAttribute('width', '%0.3f' % dx)
    doc.documentElement.setAttribute('height', '%0.3f' % dy)
    doc.documentElement.setAttribute('viewBox', '%0.3f %0.3f %0.3f %0.3f' % (x1, y1, dx, dy))
    file = open(filename, 'w')
    file.write(doc.toxml())
    file.close()