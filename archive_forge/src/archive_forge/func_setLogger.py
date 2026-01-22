import logging
from pyasn1 import __version__
from pyasn1 import error
from pyasn1.compat.octets import octs2ints
def setLogger(userLogger):
    global logger
    if userLogger:
        logger = userLogger
    else:
        logger = 0