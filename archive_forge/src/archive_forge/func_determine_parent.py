import dis
import importlib._bootstrap_external
import importlib.machinery
import marshal
import os
import io
import sys
def determine_parent(self, caller, level=-1):
    self.msgin(4, 'determine_parent', caller, level)
    if not caller or level == 0:
        self.msgout(4, 'determine_parent -> None')
        return None
    pname = caller.__name__
    if level >= 1:
        if caller.__path__:
            level -= 1
        if level == 0:
            parent = self.modules[pname]
            assert parent is caller
            self.msgout(4, 'determine_parent ->', parent)
            return parent
        if pname.count('.') < level:
            raise ImportError('relative importpath too deep')
        pname = '.'.join(pname.split('.')[:-level])
        parent = self.modules[pname]
        self.msgout(4, 'determine_parent ->', parent)
        return parent
    if caller.__path__:
        parent = self.modules[pname]
        assert caller is parent
        self.msgout(4, 'determine_parent ->', parent)
        return parent
    if '.' in pname:
        i = pname.rfind('.')
        pname = pname[:i]
        parent = self.modules[pname]
        assert parent.__name__ == pname
        self.msgout(4, 'determine_parent ->', parent)
        return parent
    self.msgout(4, 'determine_parent -> None')
    return None