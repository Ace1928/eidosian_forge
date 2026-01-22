import re
import sys
import types
import copy
import os
import inspect
def writetab(self, lextab, outputdir=''):
    if isinstance(lextab, types.ModuleType):
        raise IOError("Won't overwrite existing lextab module")
    basetabmodule = lextab.split('.')[-1]
    filename = os.path.join(outputdir, basetabmodule) + '.py'
    with open(filename, 'w') as tf:
        tf.write("# %s.py. This file automatically created by PLY (version %s). Don't edit!\n" % (basetabmodule, __version__))
        tf.write('_tabversion   = %s\n' % repr(__tabversion__))
        tf.write('_lextokens    = set(%s)\n' % repr(tuple(self.lextokens)))
        tf.write('_lexreflags   = %s\n' % repr(self.lexreflags))
        tf.write('_lexliterals  = %s\n' % repr(self.lexliterals))
        tf.write('_lexstateinfo = %s\n' % repr(self.lexstateinfo))
        tabre = {}
        for statename, lre in self.lexstatere.items():
            titem = []
            for (pat, func), retext, renames in zip(lre, self.lexstateretext[statename], self.lexstaterenames[statename]):
                titem.append((retext, _funcs_to_names(func, renames)))
            tabre[statename] = titem
        tf.write('_lexstatere   = %s\n' % repr(tabre))
        tf.write('_lexstateignore = %s\n' % repr(self.lexstateignore))
        taberr = {}
        for statename, ef in self.lexstateerrorf.items():
            taberr[statename] = ef.__name__ if ef else None
        tf.write('_lexstateerrorf = %s\n' % repr(taberr))
        tabeof = {}
        for statename, ef in self.lexstateeoff.items():
            tabeof[statename] = ef.__name__ if ef else None
        tf.write('_lexstateeoff = %s\n' % repr(tabeof))