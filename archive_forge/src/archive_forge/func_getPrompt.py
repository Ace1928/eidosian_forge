import threading, inspect, shlex
def getPrompt(self):
    return '[%s]:' % ('connected' if self.connected else 'offline')