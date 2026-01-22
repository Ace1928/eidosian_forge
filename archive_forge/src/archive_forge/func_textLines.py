from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def textLines(self, stuff, trim=1):
    """prints multi-line or newlined strings, moving down.  One
        comon use is to quote a multi-line block in your Python code;
        since this may be indented, by default it trims whitespace
        off each line and from the beginning; set trim=0 to preserve
        whitespace."""
    if isStr(stuff):
        lines = asUnicode(stuff).strip().split(u'\n')
        if trim == 1:
            lines = [s.strip() for s in lines]
    elif isinstance(stuff, (tuple, list)):
        lines = stuff
    else:
        assert 1 == 0, 'argument to textlines must be string,, list or tuple'
    for line in lines:
        self.textLine(line)