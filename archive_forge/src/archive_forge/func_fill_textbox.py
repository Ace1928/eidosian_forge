import io
import math
import os
import typing
import weakref
def fill_textbox(writer: fitz.TextWriter, rect: rect_like, text: typing.Union[str, list], pos: point_like=None, font: typing.Optional[fitz.Font]=None, fontsize: float=11, lineheight: OptFloat=None, align: int=0, warn: bool=None, right_to_left: bool=False, small_caps: bool=False) -> tuple:
    """Fill a rectangle with text.

    Args:
        writer: fitz.TextWriter object (= "self")
        rect: rect-like to receive the text.
        text: string or list/tuple of strings.
        pos: point-like start position of first word.
        font: fitz.Font object (default fitz.Font('helv')).
        fontsize: the fontsize.
        lineheight: overwrite the font property
        align: (int) 0 = left, 1 = center, 2 = right, 3 = justify
        warn: (bool) text overflow action: none, warn, or exception
        right_to_left: (bool) indicate right-to-left language.
    """
    rect = fitz.Rect(rect)
    if rect.is_empty:
        raise ValueError('fill rect must not empty.')
    if type(font) is not fitz.Font:
        font = fitz.Font('helv')

    def textlen(x):
        """Return length of a string."""
        return font.text_length(x, fontsize=fontsize, small_caps=small_caps)

    def char_lengths(x):
        """Return list of single character lengths for a string."""
        return font.char_lengths(x, fontsize=fontsize, small_caps=small_caps)

    def append_this(pos, text):
        ret = writer.append(pos, text, font=font, fontsize=fontsize, small_caps=small_caps)
        return ret
    tolerance = fontsize * 0.2
    space_len = textlen(' ')
    std_width = rect.width - tolerance
    std_start = rect.x0 + tolerance

    def norm_words(width, words):
        """Cut any word in pieces no longer than 'width'."""
        nwords = []
        word_lengths = []
        for w in words:
            wl_lst = char_lengths(w)
            wl = sum(wl_lst)
            if wl <= width:
                nwords.append(w)
                word_lengths.append(wl)
                continue
            n = len(wl_lst)
            while n > 0:
                wl = sum(wl_lst[:n])
                if wl <= width:
                    nwords.append(w[:n])
                    word_lengths.append(wl)
                    w = w[n:]
                    wl_lst = wl_lst[n:]
                    n = len(wl_lst)
                else:
                    n -= 1
        return (nwords, word_lengths)

    def output_justify(start, line):
        """Justified output of a line."""
        words = [w for w in line.split(' ') if w != '']
        nwords = len(words)
        if nwords == 0:
            return
        if nwords == 1:
            append_this(start, words[0])
            return
        tl = sum([textlen(w) for w in words])
        gaps = nwords - 1
        gapl = (std_width - tl) / gaps
        for w in words:
            _, lp = append_this(start, w)
            start.x = lp.x + gapl
        return
    asc = font.ascender
    dsc = font.descender
    if not lineheight:
        if asc - dsc <= 1:
            lheight = 1.2
        else:
            lheight = asc - dsc
    else:
        lheight = lineheight
    LINEHEIGHT = fontsize * lheight
    width = std_width
    if pos is not None:
        pos = fitz.Point(pos)
    else:
        pos = rect.tl + (tolerance, fontsize * asc)
    if pos not in rect:
        raise ValueError('Text must start in rectangle.')
    if align == fitz.TEXT_ALIGN_CENTER:
        factor = 0.5
    elif align == fitz.TEXT_ALIGN_RIGHT:
        factor = 1.0
    else:
        factor = 0
    if type(text) is str:
        textlines = text.splitlines()
    else:
        textlines = []
        for line in text:
            textlines.extend(line.splitlines())
    max_lines = int((rect.y1 - pos.y) / LINEHEIGHT) + 1
    new_lines = []
    no_justify = []
    for i, line in enumerate(textlines):
        if line in ('', ' '):
            new_lines.append((line, space_len))
            width = rect.width - tolerance
            no_justify.append(len(new_lines) - 1)
            continue
        if i == 0:
            width = rect.x1 - pos.x
        else:
            width = rect.width - tolerance
        if right_to_left:
            line = writer.clean_rtl(line)
        tl = textlen(line)
        if tl <= width:
            new_lines.append((line, tl))
            no_justify.append(len(new_lines) - 1)
            continue
        words = line.split(' ')
        words, word_lengths = norm_words(std_width, words)
        n = len(words)
        while True:
            line0 = ' '.join(words[:n])
            wl = sum(word_lengths[:n]) + space_len * (len(word_lengths[:n]) - 1)
            if wl <= width:
                new_lines.append((line0, wl))
                words = words[n:]
                word_lengths = word_lengths[n:]
                n = len(words)
                line0 = None
            else:
                n -= 1
            if len(words) == 0:
                break
    nlines = len(new_lines)
    if nlines > max_lines:
        msg = 'Only fitting %i of %i lines.' % (max_lines, nlines)
        if warn is True:
            fitz.message('Warning: ' + msg)
        elif warn is False:
            raise ValueError(msg)
    start = fitz.Point()
    no_justify += [len(new_lines) - 1]
    for i in range(max_lines):
        try:
            line, tl = new_lines.pop(0)
        except IndexError:
            if g_exceptions_verbose:
                fitz.exception_info()
            break
        if right_to_left:
            line = ''.join(reversed(line))
        if i == 0:
            start = pos
        if align == fitz.TEXT_ALIGN_JUSTIFY and i not in no_justify and (tl < std_width):
            output_justify(start, line)
            start.x = std_start
            start.y += LINEHEIGHT
            continue
        if i > 0 or pos.x == std_start:
            start.x += (width - tl) * factor
        append_this(start, line)
        start.x = std_start
        start.y += LINEHEIGHT
    return new_lines