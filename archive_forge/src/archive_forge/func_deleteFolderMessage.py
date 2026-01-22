import os
from twisted.spread import pb
def deleteFolderMessage(self, folder, name):
    if '/' in name:
        raise OSError("can only delete files in '%s' directory'" % folder)
    os.rename(os.path.join(self.directory, folder, name), os.path.join(self.rootDirectory, '.Trash', folder, name))