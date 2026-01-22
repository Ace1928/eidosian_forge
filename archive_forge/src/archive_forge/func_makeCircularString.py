from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
def makeCircularString(x, y, radius, angle, text, fontName, fontSize, inside=0, G=None, textAnchor='start'):
    """make a group with circular text in it"""
    if not G:
        G = Group()
    angle %= 360
    pi180 = pi / 180
    phi = angle * pi180
    width = stringWidth(text, fontName, fontSize)
    sig = inside and -1 or 1
    hsig = sig * 0.5
    sig90 = sig * 90
    if textAnchor != 'start':
        if textAnchor == 'middle':
            phi += sig * (0.5 * width) / radius
        elif textAnchor == 'end':
            phi += sig * float(width) / radius
        elif textAnchor == 'numeric':
            phi += sig * float(numericXShift(textAnchor, text, width, fontName, fontSize, None)) / radius
    for letter in text:
        width = stringWidth(letter, fontName, fontSize)
        beta = float(width) / radius
        h = Group()
        h.add(String(0, 0, letter, fontName=fontName, fontSize=fontSize, textAnchor='start'))
        h.translate(x + cos(phi) * radius, y + sin(phi) * radius)
        h.rotate((phi - hsig * beta) / pi180 - sig90)
        G.add(h)
        phi -= sig * beta
    return G