import formatter
import string
from types import *
import htmllib
import piddle
def demoPIL(html):
    print('be patient, this is a little slow...')
    import piddlePIL
    pc = piddlePIL.PILCanvas((800, 600), 'HTMLPiddler')
    pc.drawLine(0, 0, 100, 80, color=piddle.green)
    pc.drawRect(50, 50, 750, 550, edgeColor=piddle.pink)
    ptt = HTMLPiddler(html, (100, 80), (50, 750))
    ptt.renderOn(pc)
    pc.save(format='tif')