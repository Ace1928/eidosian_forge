import os
import reportlab
from reportlab import rl_config
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase import pdfdoc
from reportlab.lib.utils import isStr
from reportlab.lib.rl_accel import fp_str, asciiBase85Encode
from reportlab.lib.boxstuff import aspectRatioFix
def cache_imagedata(self):
    image = self.image
    if not pdfutils.cachedImageExists(image):
        pdfutils.cacheImageFile(image)
    cachedname = os.path.splitext(image)[0] + (rl_config.useA85 and '.a85' or '.bin')
    imagedata = open(cachedname, 'rb').readlines()
    imagedata = list(map(str.strip, imagedata))
    return imagedata