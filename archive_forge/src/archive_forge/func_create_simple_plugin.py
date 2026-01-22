import os
import sys
from breezy import branch, osutils, registry, tests
def create_simple_plugin(self):
    return self.create_plugin_file(b'object1 = "foo"\n\n\ndef function(a,b,c):\n    return a,b,c\n\n\nclass MyClass(object):\n    def __init__(self, a):\n      self.a = a\n\n\n')