import re
import sys
import types
import copy
import os
import inspect
def readtab(self, tabfile, fdict):
    if isinstance(tabfile, types.ModuleType):
        lextab = tabfile
    else:
        exec('import %s' % tabfile)
        lextab = sys.modules[tabfile]
    if getattr(lextab, '_tabversion', '0.0') != __tabversion__:
        raise ImportError('Inconsistent PLY version')
    self.lextokens = lextab._lextokens
    self.lexreflags = lextab._lexreflags
    self.lexliterals = lextab._lexliterals
    self.lextokens_all = self.lextokens | set(self.lexliterals)
    self.lexstateinfo = lextab._lexstateinfo
    self.lexstateignore = lextab._lexstateignore
    self.lexstatere = {}
    self.lexstateretext = {}
    for statename, lre in lextab._lexstatere.items():
        titem = []
        txtitem = []
        for pat, func_name in lre:
            titem.append((re.compile(pat, lextab._lexreflags), _names_to_funcs(func_name, fdict)))
        self.lexstatere[statename] = titem
        self.lexstateretext[statename] = txtitem
    self.lexstateerrorf = {}
    for statename, ef in lextab._lexstateerrorf.items():
        self.lexstateerrorf[statename] = fdict[ef]
    self.lexstateeoff = {}
    for statename, ef in lextab._lexstateeoff.items():
        self.lexstateeoff[statename] = fdict[ef]
    self.begin('INITIAL')