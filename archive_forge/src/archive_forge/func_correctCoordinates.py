import contextlib
import re
import xml.dom.minidom as xml
import numpy as np
from .. import debug
from .. import functions as fn
from ..parametertree import Parameter
from ..Qt import QtCore, QtGui, QtSvg, QtWidgets
from .Exporter import Exporter
def correctCoordinates(node, defs, item, options):
    for d in defs:
        if d.tagName == 'linearGradient':
            d.removeAttribute('gradientUnits')
            for coord in ('x1', 'x2', 'y1', 'y2'):
                if coord.startswith('x'):
                    denominator = item.boundingRect().width()
                else:
                    denominator = item.boundingRect().height()
                percentage = round(float(d.getAttribute(coord)) * 100 / denominator)
                d.setAttribute(coord, f'{percentage}%')
            for child in filter(lambda e: isinstance(e, xml.Element) and e.tagName == 'stop', d.childNodes):
                offset = child.getAttribute('offset')
                try:
                    child.setAttribute('offset', f'{round(float(offset) * 100)}%')
                except ValueError:
                    continue
    groups = node.getElementsByTagName('g')
    groups2 = []
    for grp in groups:
        subGroups = [grp.cloneNode(deep=False)]
        textGroup = None
        for ch in grp.childNodes[:]:
            if isinstance(ch, xml.Element):
                if textGroup is None:
                    textGroup = ch.tagName == 'text'
                if ch.tagName == 'text':
                    if textGroup is False:
                        subGroups.append(grp.cloneNode(deep=False))
                        textGroup = True
                elif textGroup is True:
                    subGroups.append(grp.cloneNode(deep=False))
                    textGroup = False
            subGroups[-1].appendChild(ch)
        groups2.extend(subGroups)
        for sg in subGroups:
            node.insertBefore(sg, grp)
        node.removeChild(grp)
    groups = groups2
    for grp in groups:
        matrix = grp.getAttribute('transform')
        match = re.match('matrix\\((.*)\\)', matrix)
        if match is None:
            vals = [1, 0, 0, 1, 0, 0]
        else:
            vals = [float(a) for a in match.groups()[0].split(',')]
        tr = np.array([[vals[0], vals[2], vals[4]], [vals[1], vals[3], vals[5]]])
        removeTransform = False
        for ch in grp.childNodes:
            if not isinstance(ch, xml.Element):
                continue
            if ch.tagName == 'polyline':
                removeTransform = True
                coords = np.array([[float(a) for a in c.split(',')] for c in ch.getAttribute('points').strip().split(' ')])
                coords = fn.transformCoordinates(tr, coords, transpose=True)
                ch.setAttribute('points', ' '.join([','.join([str(a) for a in c]) for c in coords]))
            elif ch.tagName == 'path':
                removeTransform = True
                newCoords = ''
                oldCoords = ch.getAttribute('d').strip()
                if oldCoords == '':
                    continue
                for c in oldCoords.split(' '):
                    x, y = c.split(',')
                    if x[0].isalpha():
                        t = x[0]
                        x = x[1:]
                    else:
                        t = ''
                    nc = fn.transformCoordinates(tr, np.array([[float(x), float(y)]]), transpose=True)
                    newCoords += t + str(nc[0, 0]) + ',' + str(nc[0, 1]) + ' '
                if newCoords[0] != 'M':
                    newCoords = f'M{newCoords[1:]}'
                ch.setAttribute('d', newCoords)
            elif ch.tagName == 'text':
                removeTransform = False
                families = ch.getAttribute('font-family').split(',')
                if len(families) == 1:
                    font = QtGui.QFont(families[0].strip('" '))
                    if font.styleHint() == font.StyleHint.SansSerif:
                        families.append('sans-serif')
                    elif font.styleHint() == font.StyleHint.Serif:
                        families.append('serif')
                    elif font.styleHint() == font.StyleHint.Courier:
                        families.append('monospace')
                    ch.setAttribute('font-family', ', '.join([f if ' ' not in f else '"%s"' % f for f in families]))
            if removeTransform and ch.getAttribute('vector-effect') != 'non-scaling-stroke' and (grp.getAttribute('stroke-width') != ''):
                w = float(grp.getAttribute('stroke-width'))
                s = fn.transformCoordinates(tr, np.array([[w, 0], [0, 0]]), transpose=True)
                w = ((s[0] - s[1]) ** 2).sum() ** 0.5
                ch.setAttribute('stroke-width', str(w))
            if options.get('scaling stroke') is True and ch.getAttribute('vector-effect') == 'non-scaling-stroke':
                ch.removeAttribute('vector-effect')
        if removeTransform:
            grp.removeAttribute('transform')