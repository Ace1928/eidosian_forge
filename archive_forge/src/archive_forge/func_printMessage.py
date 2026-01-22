import sys
from twisted.internet import reactor
from twisted.internet.protocol import Factory
from twisted.protocols import basic
def printMessage(msg):
    print('Server Starting in %s mode' % msg)