import os
from twisted.spread import pb
def getCurMessages(self):
    return os.listdir(os.path.join(self.directory, 'cur'))