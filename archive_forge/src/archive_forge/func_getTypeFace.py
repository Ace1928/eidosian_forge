import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
def getTypeFace(faceName):
    """Lazily construct known typefaces if not found"""
    try:
        return _typefaces[faceName]
    except KeyError:
        if faceName in standardFonts:
            face = TypeFace(faceName)
            face.familyName, face.bold, face.italic = _fontdata.standardFontAttributes[faceName]
            registerTypeFace(face)
            return face
        else:
            afm = bruteForceSearchForAFM(faceName)
            if afm:
                for e in ('.pfb', '.PFB'):
                    pfb = os.path.splitext(afm)[0] + e
                    if rl_isfile(pfb):
                        break
                assert rl_isfile(pfb), 'file %s not found!' % pfb
                face = EmbeddedType1Face(afm, pfb)
                registerTypeFace(face)
                return face
            else:
                raise