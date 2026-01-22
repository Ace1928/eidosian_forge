from binascii import unhexlify
from math import ceil
from typing import Any, Dict, List, Tuple, Union, cast
from ._codecs import adobe_glyphs, charset_encoding
from ._utils import b_, logger_error, logger_warning
from .generic import (
def compute_space_width(ft: DictionaryObject, space_code: int, space_width: float) -> float:
    sp_width: float = space_width * 2.0
    w = []
    w1 = {}
    st: int = 0
    if '/DescendantFonts' in ft:
        ft1 = ft['/DescendantFonts'][0].get_object()
        try:
            w1[-1] = cast(float, ft1['/DW'])
        except Exception:
            w1[-1] = 1000.0
        if '/W' in ft1:
            w = list(ft1['/W'])
        else:
            w = []
        while len(w) > 0:
            st = w[0] if isinstance(w[0], int) else w[0].get_object()
            second = w[1].get_object()
            if isinstance(second, int):
                for x in range(st, second):
                    w1[x] = w[2]
                w = w[3:]
            elif isinstance(second, list):
                for y in second:
                    w1[st] = y
                    st += 1
                w = w[2:]
            else:
                logger_warning('unknown widths : \n' + ft1['/W'].__repr__(), __name__)
                break
        try:
            sp_width = w1[space_code]
        except Exception:
            sp_width = w1[-1] / 2.0
    elif '/Widths' in ft:
        w = list(ft['/Widths'])
        try:
            st = cast(int, ft['/FirstChar'])
            en: int = cast(int, ft['/LastChar'])
            if st > space_code or en < space_code:
                raise Exception('Not in range')
            if w[space_code - st] == 0:
                raise Exception('null width')
            sp_width = w[space_code - st]
        except Exception:
            if '/FontDescriptor' in ft and '/MissingWidth' in cast(DictionaryObject, ft['/FontDescriptor']):
                sp_width = ft['/FontDescriptor']['/MissingWidth']
            else:
                m = 0
                cpt = 0
                for x in w:
                    if x > 0:
                        m += x
                        cpt += 1
                sp_width = m / max(1, cpt) / 2
    if isinstance(sp_width, IndirectObject):
        obj = sp_width.get_object()
        if obj is None or isinstance(obj, NullObject):
            return 0.0
        return obj
    return sp_width