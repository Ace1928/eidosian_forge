from string import whitespace
from operator import truth
from unicodedata import category
from reportlab.pdfbase.pdfmetrics import stringWidth, getAscentDescent
from reportlab.platypus.paraparser import ParaParser, _PCT, _num as _parser_num, _re_us_value
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import Color
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.geomutils import normalizeTRBL
from reportlab.lib.textsplit import wordSplit, ALL_CANNOT_START
from reportlab.lib.styles import ParagraphStyle
from copy import deepcopy
from reportlab.lib.abag import ABag
from reportlab.rl_config import decimalSymbol, _FUZZ, paraFontSizeHeightOffset,\
from reportlab.lib.utils import _className, isBytes, isStr
from reportlab.lib.rl_accel import sameFrag
import re
from types import MethodType
def drawPara(self, debug=0):
    """Draws a paragraph according to the given style.
        Returns the final y position at the bottom. Not safe for
        paragraphs without spaces e.g. Japanese; wrapping
        algorithm will go infinite."""
    canvas = self.canv
    style = self.style
    blPara = self.blPara
    lines = blPara.lines
    leading = style.leading
    autoLeading = getattr(self, 'autoLeading', getattr(style, 'autoLeading', ''))
    leftIndent = style.leftIndent
    cur_x = leftIndent
    if debug:
        bw = 0.5
        bc = Color(1, 1, 0)
        bg = Color(0.9, 0.9, 0.9)
    else:
        bw = getattr(style, 'borderWidth', None)
        bc = getattr(style, 'borderColor', None)
        bg = style.backColor
    if bg or (bc and bw):
        canvas.saveState()
        op = canvas.rect
        kwds = dict(fill=0, stroke=0)
        if bc and bw:
            canvas.setStrokeColor(bc)
            canvas.setLineWidth(bw)
            kwds['stroke'] = 1
            br = getattr(style, 'borderRadius', 0)
            if br and (not debug):
                op = canvas.roundRect
                kwds['radius'] = br
        if bg:
            canvas.setFillColor(bg)
            kwds['fill'] = 1
        bp = getattr(style, 'borderPadding', 0)
        tbp, rbp, bbp, lbp = normalizeTRBL(bp)
        op(leftIndent - lbp, -bbp, self.width - (leftIndent + style.rightIndent) + lbp + rbp, self.height + tbp + bbp, **kwds)
        canvas.restoreState()
    nLines = len(lines)
    bulletText = self.bulletText
    if nLines > 0:
        _offsets = getattr(self, '_offsets', [0])
        _offsets += (nLines - len(_offsets)) * [_offsets[-1]]
        canvas.saveState()
        alignment = style.alignment
        offset = style.firstLineIndent + _offsets[0]
        lim = nLines - 1
        noJustifyLast = not getattr(self, '_JustifyLast', False)
        jllwc = style.justifyLastLine
        isRTL = style.wordWrap == 'RTL'
        bRTL = isRTL and self._wrapWidths or False
        if blPara.kind == 0:
            if alignment == TA_LEFT:
                dpl = _leftDrawParaLine
            elif alignment == TA_CENTER:
                dpl = _centerDrawParaLine
            elif alignment == TA_RIGHT:
                dpl = _rightDrawParaLine
            elif alignment == TA_JUSTIFY:
                dpl = _justifyDrawParaLineRTL if isRTL else _justifyDrawParaLine
            f = blPara
            if paraFontSizeHeightOffset:
                cur_y = self.height - f.fontSize
            else:
                cur_y = self.height - getattr(f, 'ascent', f.fontSize)
            if bulletText:
                offset = _drawBullet(canvas, offset, cur_y, bulletText, style, rtl=bRTL)
            canvas.setFillColor(f.textColor)
            tx = self.beginText(cur_x, cur_y)
            tx.preformatted = 'preformatted' in self.__class__.__name__.lower()
            if autoLeading == 'max':
                leading = max(leading, blPara.ascent - blPara.descent)
            elif autoLeading == 'min':
                leading = blPara.ascent - blPara.descent
            tx.direction = self.style.wordWrap
            tx.setFont(f.fontName, f.fontSize, leading)
            ws = lines[0][0]
            words = lines[0][1]
            lastLine = noJustifyLast and nLines == 1
            if lastLine and jllwc and (len(words) > jllwc):
                lastLine = False
            t_off = dpl(tx, offset, ws, words, lastLine)
            if f.us_lines or f.link:
                tx._do_line = MethodType(_do_line, tx)
                tx.xs = xs = tx.XtraState = ABag()
                _setTXLineProps(tx, canvas, style)
                xs.cur_y = cur_y
                xs.f = f
                xs.style = style
                xs.lines = lines
                xs.link = f.link
                xs.textColor = f.textColor
                xs.backColors = []
                dx = t_off + leftIndent
                if alignment != TA_JUSTIFY or lastLine:
                    ws = 0
                if f.us_lines:
                    _do_under_line(0, dx, ws, tx, f.us_lines)
                if f.link:
                    _do_link_line(0, dx, ws, tx)
                for i in range(1, nLines):
                    ws = lines[i][0]
                    words = lines[i][1]
                    lastLine = noJustifyLast and i == lim
                    if lastLine and jllwc and (len(words) > jllwc):
                        lastLine = False
                    t_off = dpl(tx, _offsets[i], ws, words, lastLine)
                    dx = t_off + leftIndent
                    if alignment != TA_JUSTIFY or lastLine:
                        ws = 0
                    if f.us_lines:
                        _do_under_line(i, t_off, ws, tx, f.us_lines)
                    if f.link:
                        _do_link_line(i, dx, ws, tx)
            else:
                for i in range(1, nLines):
                    words = lines[i][1]
                    lastLine = noJustifyLast and i == lim
                    if lastLine and jllwc and (len(words) > jllwc):
                        lastLine = False
                    dpl(tx, _offsets[i], lines[i][0], words, lastLine)
        else:
            if isRTL:
                for line in lines:
                    line.words = line.words[::-1]
            f = lines[0]
            if paraFontSizeHeightOffset:
                cur_y = self.height - f.fontSize
            else:
                cur_y = self.height - getattr(f, 'ascent', f.fontSize)
            dpl = _leftDrawParaLineX
            if bulletText:
                oo = offset
                offset = _drawBullet(canvas, offset, cur_y, bulletText, style, rtl=bRTL)
            if alignment == TA_LEFT:
                dpl = _leftDrawParaLineX
            elif alignment == TA_CENTER:
                dpl = _centerDrawParaLineX
            elif alignment == TA_RIGHT:
                dpl = _rightDrawParaLineX
            elif alignment == TA_JUSTIFY:
                dpl = _justifyDrawParaLineXRTL if isRTL else _justifyDrawParaLineX
            else:
                raise ValueError('bad align %s' % repr(alignment))
            tx = self.beginText(cur_x, cur_y)
            tx.preformatted = 'preformatted' in self.__class__.__name__.lower()
            _setTXLineProps(tx, canvas, style)
            tx._do_line = MethodType(_do_line, tx)
            tx.direction = self.style.wordWrap
            xs = tx.XtraState = ABag()
            xs.textColor = None
            xs.backColor = None
            xs.rise = 0
            xs.backColors = []
            xs.us_lines = {}
            xs.links = {}
            xs.link = {}
            xs.leading = style.leading
            xs.leftIndent = leftIndent
            tx._leading = None
            tx._olb = None
            xs.cur_y = cur_y
            xs.f = f
            xs.style = style
            xs.autoLeading = autoLeading
            xs.paraWidth = self.width
            tx._fontname, tx._fontsize = (None, None)
            line = lines[0]
            lastLine = noJustifyLast and nLines == 1
            if lastLine and jllwc and (line.wordCount > jllwc):
                lastLine = False
            dpl(tx, offset, line, lastLine)
            _do_post_text(tx)
            for i in range(1, nLines):
                line = lines[i]
                lastLine = noJustifyLast and i == lim
                if lastLine and jllwc and (line.wordCount > jllwc):
                    lastLine = False
                dpl(tx, _offsets[i], line, lastLine)
                _do_post_text(tx)
        canvas.drawText(tx)
        canvas.restoreState()