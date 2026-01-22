from fontTools.misc import psCharStrings
from fontTools import ttLib
from fontTools.pens.basePen import NullPen
from fontTools.misc.roundTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.subset.util import _add_method, _uniq_sort
@_add_method(ttLib.getTableClass('CFF '))
def remove_hints(self):
    cff = self.cff
    for fontname in cff.keys():
        font = cff[fontname]
        cs = font.CharStrings
        css = set()
        for g in font.charset:
            c, _ = cs.getItemAndSelector(g)
            c.decompile()
            subrs = getattr(c.private, 'Subrs', [])
            decompiler = _DehintingT2Decompiler(css, subrs, c.globalSubrs, c.private.nominalWidthX, c.private.defaultWidthX, c.private)
            decompiler.execute(c)
            c.width = decompiler.width
        for charstring in css:
            charstring.drop_hints()
        del css
        all_privs = []
        if hasattr(font, 'FDArray'):
            all_privs.extend((fd.Private for fd in font.FDArray))
        else:
            all_privs.append(font.Private)
        for priv in all_privs:
            for k in ['BlueValues', 'OtherBlues', 'FamilyBlues', 'FamilyOtherBlues', 'BlueScale', 'BlueShift', 'BlueFuzz', 'StemSnapH', 'StemSnapV', 'StdHW', 'StdVW', 'ForceBold', 'LanguageGroup', 'ExpansionFactor']:
                if hasattr(priv, k):
                    setattr(priv, k, None)
    self.remove_unused_subroutines()