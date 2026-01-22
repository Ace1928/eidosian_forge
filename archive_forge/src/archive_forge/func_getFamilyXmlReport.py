import sys, os, pickle
from hashlib import md5
from xml.sax.saxutils import quoteattr
from time import process_time as clock
from reportlab.lib.utils import asBytes, asNative as _asNative
from reportlab.lib.utils import rl_isdir, rl_isfile, rl_listdir, rl_getmtime
def getFamilyXmlReport(self):
    """Reports on all families found as XML.
        """
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
    lines.append('<font_families>')
    for dirName in self._dirs:
        lines.append('    <directory name=%s/>' % quoteattr(asNative(dirName)))
    for familyName in self.getFamilyNames():
        if familyName:
            lines.append('    <family name=%s>' % quoteattr(asNative(familyName)))
            for font in self.getFontsInFamily(familyName):
                lines.append('        ' + font.getTag())
            lines.append('    </family>')
    lines.append('</font_families>')
    return '\n'.join(lines)