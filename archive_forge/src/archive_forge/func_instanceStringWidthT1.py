import reportlab
def instanceStringWidthT1(self, text, size, encoding='utf8'):
    """This is the "purist" approach to width"""
    if not isUnicode(text):
        text = text.decode(encoding)
    return sum((sum(map(f.widths.__getitem__, t)) for f, t in unicode2T1(text, [self] + self.substitutionFonts))) * 0.001 * size