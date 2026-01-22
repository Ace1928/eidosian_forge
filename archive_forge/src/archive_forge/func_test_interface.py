import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def test_interface(self):
    wrong_arch = 'ppc64' if self.arch != 'ppc64' else 'x86'
    wrong_cc = 'clang' if self.cc != 'clang' else 'icc'
    opt = self.opt()
    assert_(getattr(opt, 'cc_on_' + self.arch))
    assert_(not getattr(opt, 'cc_on_' + wrong_arch))
    assert_(getattr(opt, 'cc_is_' + self.cc))
    assert_(not getattr(opt, 'cc_is_' + wrong_cc))