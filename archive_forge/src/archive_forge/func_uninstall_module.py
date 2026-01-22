from lxml import etree
import sys
import re
import doctest
def uninstall_module(self):
    if self.del_module:
        import sys
        del sys.modules[self.del_module]
        if '.' in self.del_module:
            package, module = self.del_module.rsplit('.', 1)
            package_mod = sys.modules[package]
            delattr(package_mod, module)