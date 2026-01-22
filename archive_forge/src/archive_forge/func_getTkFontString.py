import tkFont
import Tkinter
import rdkit.sping.pid
def getTkFontString(self, font):
    """Return a string suitable to pass as the -font option to
        to a Tk widget based on the piddle-style FONT"""
    tkfont = self.piddleToTkFont(font)
    return '-family %(family)s -size %(size)s -weight %(weight)s -slant %(slant)s -underline %(underline)s' % tkfont.config()