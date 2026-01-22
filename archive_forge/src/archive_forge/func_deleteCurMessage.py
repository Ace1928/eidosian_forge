import os
from twisted.spread import pb
def deleteCurMessage(self, name):
    return self.deleteFolderMessage('cur', name)