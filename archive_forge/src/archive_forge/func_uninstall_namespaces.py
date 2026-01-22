import os
from distutils import log
import itertools
def uninstall_namespaces(self):
    filename = self._get_nspkg_file()
    if not os.path.exists(filename):
        return
    log.info('Removing %s', filename)
    os.remove(filename)