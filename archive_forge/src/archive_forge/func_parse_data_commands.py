import bisect
import sys
import logging
import os
import os.path
import ply.lex as lex
import ply.yacc as yacc
from inspect import getfile, currentframe
from pyomo.common.fileutils import this_file
from pyomo.core.base.util import flatten_tuple
def parse_data_commands(data=None, filename=None, debug=0, outputdir=None):
    global dat_lexer
    global dat_yaccer
    global dat_yaccer_tabfile
    if outputdir is None:
        outputdir = os.path.dirname(getfile(currentframe()))
        _tabfile = os.path.join(outputdir, tabmodule + '.py')
        if not os.access(outputdir, os.W_OK):
            _file = this_file()
            logger = logging.getLogger('pyomo.dataportal')
            if os.path.exists(_tabfile) and os.path.getmtime(_file) >= os.path.getmtime(_tabfile):
                logger.warning('Potentially outdated DAT Parse Table found in source tree (%s), but you do not have write access to that directory, so we cannot update it.  Please notify you system administrator to remove that file' % (_tabfile,))
            if os.path.exists(_tabfile + 'c') and os.path.getmtime(_file) >= os.path.getmtime(_tabfile + 'c'):
                logger.warning('Potentially outdated DAT Parse Table found in source tree (%s), but you do not have write access to that directory, so we cannot update it.  Please notify you system administrator to remove that file' % (_tabfile + 'c',))
            outputdir = os.getcwd()
    if dat_lexer is None:
        _parser_out = os.path.join(outputdir, 'parser.out')
        if os.path.exists(_parser_out):
            os.remove(_parser_out)
        _tabfile = dat_yaccer_tabfile = os.path.join(outputdir, tabmodule + '.py')
        if debug > 0 or (os.path.exists(_tabfile) and os.path.getmtime(__file__) >= os.path.getmtime(_tabfile)):
            if os.path.exists(_tabfile):
                os.remove(_tabfile)
            if os.path.exists(_tabfile + 'c'):
                os.remove(_tabfile + 'c')
            for _mod in list(sys.modules.keys()):
                if _mod == tabmodule or _mod.endswith('.' + tabmodule):
                    del sys.modules[_mod]
        dat_lexer = lex.lex()
        tmpsyspath = sys.path
        sys.path.append(outputdir)
        dat_yaccer = yacc.yacc(debug=debug, tabmodule=tabmodule, outputdir=outputdir, optimize=True)
        sys.path = tmpsyspath
    dat_lexer.linepos = []
    global _parse_info
    _parse_info = {}
    _parse_info[None] = []
    if filename is not None:
        if data is not None:
            raise ValueError('parse_data_commands: cannot specify both data and filename arguments')
        with open(filename, 'r') as FILE:
            data = FILE.read()
    if data is None:
        return None
    dat_yaccer.parse(data, lexer=dat_lexer, debug=debug)
    return _parse_info