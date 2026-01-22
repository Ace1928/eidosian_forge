import reportlab
def instanceStringWidthTTF(self, text, size, encoding='utf-8'):
    """Calculate text width"""
    if not isUnicode(text):
        text = text.decode(encoding or 'utf-8')
    g = self.face.charWidths.get
    dw = self.face.defaultWidth
    return 0.001 * size * sum((g(ord(u), dw) for u in text))