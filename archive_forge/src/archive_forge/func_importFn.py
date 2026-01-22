import importlib
import logging
def importFn(self, what=None):
    imp = self.modulename if not what else '%s.%s' % (self.modulename, what)
    return importlib.import_module(imp)