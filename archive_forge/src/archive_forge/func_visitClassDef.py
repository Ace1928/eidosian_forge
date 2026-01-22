from __future__ import with_statement
import optparse
import sys
import tokenize
from collections import defaultdict
def visitClassDef(self, node):
    old_classname = self.classname
    self.classname += node.name + '.'
    self.dispatch_list(node.body)
    self.classname = old_classname