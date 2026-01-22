import os
from twisted.spread import pb
def getNewMessage(self, name):
    return self.getFolderMessage('new', name)