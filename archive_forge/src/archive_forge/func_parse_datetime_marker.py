import calendar
import datetime
import decimal
import re
from typing import Any, Iterator, List, Optional, Tuple, Union
from unicodedata import category
from ..exceptions import xpath_error
from ..regex import translate_pattern
from ._translation_maps import ALPHABET_CHARACTERS, OTHER_NUMBERS, ROMAN_NUMERALS_MAP, \
def parse_datetime_marker(marker: str, dt: datetime.datetime, lang: Optional[str]=None) -> str:
    min_width: int
    max_width: Optional[int]
    component = marker[1]
    fmt_token = marker[2:-1]
    if ',' not in fmt_token:
        presentation, width = (fmt_token, '')
    else:
        presentation, width = fmt_token.rsplit(',', maxsplit=1)
    if not presentation:
        fmt_modifier = ''
        if component in 'Hhf':
            presentation = '1'
        elif component in 'ms':
            presentation = '01'
        elif component in 'Zz':
            presentation = '01:01'
        else:
            presentation = 'n'
    elif presentation == 'a':
        fmt_modifier = ''
    else:
        _match = FMT_MODIFIER_PATTERN.search(presentation)
        if _match is None:
            fmt_modifier = ''
        else:
            fmt_modifier = _match.group(0)
            if fmt_modifier:
                presentation = presentation[:-len(fmt_modifier)]
        if presentation.startswith('#') and presentation.endswith('#'):
            msg_tmpl = 'Invalid formatting component {!r}'
            raise xpath_error('FOFD1340', msg_tmpl.format(component))
    for pch in presentation:
        if pch.isdigit():
            zero_cp = ord(pch) - int(pch)
            zero_ch = chr(zero_cp)
            break
    else:
        zero_cp, zero_ch = (ord('0'), '0')
    digits = sum((c.isdigit() for c in presentation))
    opt_digits = presentation.count('#')
    if not width or width == '*':
        if digits > 1:
            min_width, max_width = (digits, digits + opt_digits)
        else:
            min_width, max_width = (0, None)
    else:
        min_width, max_width = parse_width(width)
        if digits > 1:
            min_width = max(min_width, digits)
            if max_width:
                max_width = max(max_width, digits + opt_digits)
    if component == 'Y':
        value = str(abs(dt.year))
    elif component == 'M':
        if presentation.lower().startswith('n') and lang is not None:
            value = int_to_month(dt.month, lang)
        else:
            value = str(dt.month)
    elif component == 'D':
        value = str(dt.day)
    elif component == 'H':
        value = str(dt.hour)
    elif component == 'h':
        if dt.hour == 0:
            value = '12'
        elif dt.hour > 12:
            value = str(dt.hour % 12)
        else:
            value = str(dt.hour)
    elif component == 'P':
        value = 'a.m.' if dt.hour < 12 else 'p.m.'
    elif component == 'm':
        value = str(dt.minute)
    elif component == 's':
        value = str(dt.second)
    elif component == 'f':
        value = str('{:06}'.format(dt.microsecond))
    elif component == 'z' or component == 'Z':
        if presentation == 'N':
            value = dt.tzname() or ''
        elif dt.tzinfo is None:
            value = '+00:00'
        else:
            value = str(dt)
            if value.endswith('Z'):
                value = '+00:00'
            else:
                value = value[-6:]
    elif component == 'W':
        value = str(dt.isocalendar()[1])
    elif component == 'w':
        value = str(week_in_month(dt))
    elif component == 'F':
        if presentation.lower().startswith('n') and lang is not None:
            value = int_to_weekday(dt.isocalendar()[2], lang)
        else:
            value = str(dt.isocalendar()[2])
    elif component == 'E':
        if dt.year < 0:
            value = 'BC'
        else:
            value = 'AD'
    elif component == 'd':
        delta = dt - type(dt)(dt.year, 1, 1)
        value = str(1 + delta.seconds // 86400)
    else:
        msg_tmpl = 'Invalid formatting component {!r}'
        raise xpath_error('FOFD1340', msg_tmpl.format(component))
    sign = ''
    left_to_right = component != 'Y'
    if presentation == 'n':
        fmt_chunk = value.lower()
    elif presentation == 'N':
        fmt_chunk = value.upper()
    elif presentation == 'Nn':
        fmt_chunk = value.title()
    elif presentation == 'I' or presentation == 'i':
        fmt_chunk = value
    elif presentation == 'Z' and component == 'Z':
        if dt.tzinfo is None:
            fmt_chunk = MILITARY_TIME_ZONES[None]
        elif value.endswith(':00'):
            fmt_chunk = MILITARY_TIME_ZONES.get(value[:3], value)
        else:
            fmt_chunk = value
    elif presentation == 'w':
        fmt_chunk = int_to_words(int(value), lang, fmt_modifier)
    elif presentation == 'W':
        fmt_chunk = int_to_words(int(value), lang, fmt_modifier).upper()
    elif presentation == 'Ww':
        fmt_chunk = int_to_words(int(value), lang, fmt_modifier).title()
    elif presentation == 'a':
        fmt_chunk = int_to_alphabetic(int(value), lang)
    elif presentation == 'A':
        fmt_chunk = int_to_alphabetic(int(value), lang).upper()
    else:
        left_to_right = False
        k = 0
        pch = ''
        chars = []
        if value.startswith('-') or value.startswith('+'):
            sign = value[0]
            value = value[1:]
        if component in 'zZ':
            if presentation.isdigit():
                if len(presentation) <= 2:
                    if value.endswith(':00'):
                        value = value[:-3]
                        left_to_right = True
                    elif len(presentation) == 1:
                        presentation = '#0:01'
                        min_width, max_width = (3, 4)
                    else:
                        presentation = '01:01'
                        min_width = max_width = 4
                elif len(presentation) == 3:
                    presentation = '#001'
                    min_width, max_width = (3, 4)
            elif presentation.replace(':', '', 1).isdigit():
                if len(presentation) == 4:
                    presentation = '#0:01'
                    min_width, max_width = (3, 4)
        if component != 'f':
            presentation = ''.join(reversed(presentation))
            value = ''.join(reversed(value))
        for ch in value:
            try:
                pch = presentation[k]
            except IndexError:
                if ch == '0' and (not pch.isdigit()):
                    break
            else:
                k += 1
            while pch != '#' and (not pch.isdigit()):
                chars.append(pch)
                min_width += 1
                if max_width is not None:
                    max_width += 1
                try:
                    pch = presentation[k]
                except IndexError:
                    break
                else:
                    k += 1
            else:
                if ch.isdigit():
                    chars.append(ch)
        if component != 'f':
            fmt_chunk = ''.join(reversed(chars))
        else:
            fmt_chunk = ''.join(chars)
        if 'o' in fmt_modifier:
            try:
                fmt_chunk += ordinal_suffix(int(fmt_chunk))
            except ValueError:
                pass
            else:
                min_width += 2
                if max_width is not None:
                    max_width += 2
    if len(fmt_chunk) < min_width and component not in 'PzZ':
        if component in 'f':
            fmt_chunk += zero_ch * (min_width - len(fmt_chunk))
        else:
            fmt_chunk = zero_ch * (min_width - len(fmt_chunk)) + fmt_chunk
    if max_width:
        if left_to_right or component in 'f':
            fmt_chunk = fmt_chunk[:max_width]
        else:
            fmt_chunk = fmt_chunk[max(0, len(fmt_chunk) - max_width):]
    if component in 'zZ':
        if not min_width:
            fmt_chunk = fmt_chunk.lstrip('0')
            if not fmt_chunk:
                return 'Z' if component == 'Z' else 'GMT' + sign + '0'
        else:
            try:
                nz_first = min((k for k in range(len(fmt_chunk)) if fmt_chunk[k] != zero_ch))
            except ValueError:
                fmt_chunk = fmt_chunk[max(0, len(fmt_chunk) - min_width):]
            else:
                fmt_chunk = fmt_chunk[max(0, min(nz_first, len(fmt_chunk) - min_width)):]
    elif min_width == 3 and component == 'F':
        fmt_chunk = fmt_chunk[:3]
    elif min_width or component == 'f':
        try:
            nz_last = max((k for k in range(len(fmt_chunk)) if fmt_chunk[k] != zero_ch))
        except ValueError:
            nz_last = 0
        fmt_chunk = fmt_chunk[:max(min_width, nz_last + 1)]
    if zero_ch != '0':
        fmt_chunk = ''.join((chr(zero_cp + int(ch)) if ch.isdigit() else ch for ch in fmt_chunk))
    if component == 'z':
        return 'GMT' + sign + fmt_chunk
    if presentation == 'I':
        return sign + int_to_roman(int(fmt_chunk))
    elif presentation == 'i':
        return sign + int_to_roman(int(fmt_chunk)).lower()
    return sign + fmt_chunk