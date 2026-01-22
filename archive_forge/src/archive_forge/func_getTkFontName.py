import tkFont
import Tkinter
import rdkit.sping.pid
def getTkFontName(self, font):
    """Return a name associated with the piddle-style FONT"""
    tkfont = self.piddleToTkFont(font)
    return str(tkfont)