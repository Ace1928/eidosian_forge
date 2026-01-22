import sys
import pkg_resources
import inspect
import logging
def find_module_for_extension(self, name):
    for ep in pkg_resources.iter_entry_points('pecan.extension'):
        if ep.name != name:
            continue
        log.debug('%s loading extension %s', self.__class__.__name__, ep)
        module = ep.load()
        if not inspect.ismodule(module):
            log.debug('%s is not a module, skipping...' % module)
            continue
        return module
    raise PecanExtensionMissing('The `pecan.ext.%s` extension is not installed.' % name)