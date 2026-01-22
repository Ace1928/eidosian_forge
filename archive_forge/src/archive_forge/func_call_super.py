from lxml import etree
import sys
import re
import doctest
def call_super(self, *args, **kw):
    self.uninstall_clone()
    try:
        return self.check_func(*args, **kw)
    finally:
        self.install_clone()