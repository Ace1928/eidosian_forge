imports, including parts of the standard library and installed
import glob
import imp
import os
import sys
from zipimport import zipimporter, ZipImportError
def knows(self, fullname):
    return fullname in self._libs