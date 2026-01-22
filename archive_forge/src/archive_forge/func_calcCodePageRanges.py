from fontTools.misc import sstruct
from fontTools.misc.roundTools import otRound
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.ttLib.tables import DefaultTable
import bisect
import logging
def calcCodePageRanges(unicodes):
    """Given a set of Unicode codepoints (integers), calculate the
    corresponding OS/2 CodePage range bits.
    This is a direct translation of FontForge implementation:
    https://github.com/fontforge/fontforge/blob/7b2c074/fontforge/tottf.c#L3158
    """
    bits = set()
    hasAscii = set(range(32, 126)).issubset(unicodes)
    hasLineart = ord('┤') in unicodes
    for uni in unicodes:
        if uni == ord('Þ') and hasAscii:
            bits.add(0)
        elif uni == ord('Ľ') and hasAscii:
            bits.add(1)
            if hasLineart:
                bits.add(58)
        elif uni == ord('Б'):
            bits.add(2)
            if ord('Ѕ') in unicodes and hasLineart:
                bits.add(57)
            if ord('╜') in unicodes and hasLineart:
                bits.add(49)
        elif uni == ord('Ά'):
            bits.add(3)
            if hasLineart and ord('½') in unicodes:
                bits.add(48)
            if hasLineart and ord('√') in unicodes:
                bits.add(60)
        elif uni == ord('İ') and hasAscii:
            bits.add(4)
            if hasLineart:
                bits.add(56)
        elif uni == ord('א'):
            bits.add(5)
            if hasLineart and ord('√') in unicodes:
                bits.add(53)
        elif uni == ord('ر'):
            bits.add(6)
            if ord('√') in unicodes:
                bits.add(51)
            if hasLineart:
                bits.add(61)
        elif uni == ord('ŗ') and hasAscii:
            bits.add(7)
            if hasLineart:
                bits.add(59)
        elif uni == ord('₫') and hasAscii:
            bits.add(8)
        elif uni == ord('ๅ'):
            bits.add(16)
        elif uni == ord('エ'):
            bits.add(17)
        elif uni == ord('ㄅ'):
            bits.add(18)
        elif uni == ord('ㄱ'):
            bits.add(19)
        elif uni == ord('央'):
            bits.add(20)
        elif uni == ord('곴'):
            bits.add(21)
        elif uni == ord('♥') and hasAscii:
            bits.add(30)
        elif uni == ord('þ') and hasAscii and hasLineart:
            bits.add(54)
        elif uni == ord('╚') and hasAscii:
            bits.add(62)
            bits.add(63)
        elif hasAscii and hasLineart and (ord('√') in unicodes):
            if uni == ord('Å'):
                bits.add(50)
            elif uni == ord('é'):
                bits.add(52)
            elif uni == ord('õ'):
                bits.add(55)
    if hasAscii and ord('‰') in unicodes and (ord('∑') in unicodes):
        bits.add(29)
    return bits