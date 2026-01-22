from fontTools.misc.loggingTools import deprecateFunction
import logging
from fontTools.ttLib.ttFont import *
from fontTools.ttLib.ttCollection import TTCollection
@deprecateFunction('use logging instead', category=DeprecationWarning)
def debugmsg(msg):
    import time
    print(msg + time.strftime('  (%H:%M:%S)', time.localtime(time.time())))