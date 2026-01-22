import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
def makePDFObject(self):
    """Returns a PDF Object representing self"""
    from reportlab.pdfbase import pdfdoc
    D = {}
    baseEncodingName = self.baseEncodingName
    baseEnc = getEncoding(baseEncodingName)
    differences = self.getDifferences(baseEnc)
    if differences == []:
        return pdfdoc.PDFName(baseEncodingName)
    else:
        diffArray = []
        for range in differences:
            diffArray.append(range[0])
            for glyphName in range[1:]:
                if glyphName is not None:
                    diffArray.append('/' + glyphName)
        D['Differences'] = pdfdoc.PDFArray(diffArray)
        if baseEncodingName in ('MacRomanEncoding', 'MacExpertEncoding', 'WinAnsiEncoding'):
            D['BaseEncoding'] = pdfdoc.PDFName(baseEncodingName)
        D['Type'] = pdfdoc.PDFName('Encoding')
        PD = pdfdoc.PDFDictionary(D)
        return PD