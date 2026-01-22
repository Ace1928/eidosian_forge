from sympy.external.gmpy import GROUND_TYPES
from sympy.external.importtools import version_tuple
from sympy.interactive.printing import init_printing
from sympy.utilities.misc import ARCH
from sympy import *
def enable_automatic_symbols(shell):
    """Allow IPython to automatically create symbols (``isympy -a``). """
    import re
    re_nameerror = re.compile("name '(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)' is not defined")

    def _handler(self, etype, value, tb, tb_offset=None):
        """Handle :exc:`NameError` exception and allow injection of missing symbols. """
        if etype is NameError and tb.tb_next and (not tb.tb_next.tb_next):
            match = re_nameerror.match(str(value))
            if match is not None:
                self.run_cell("%(symbol)s = Symbol('%(symbol)s')" % {'symbol': match.group('symbol')}, store_history=False)
                try:
                    code = self.user_ns['In'][-1]
                except (KeyError, IndexError):
                    pass
                else:
                    self.run_cell(code, store_history=False)
                    return None
                finally:
                    self.run_cell('del %s' % match.group('symbol'), store_history=False)
        stb = self.InteractiveTB.structured_traceback(etype, value, tb, tb_offset=tb_offset)
        self._showtraceback(etype, value, stb)
    shell.set_custom_exc((NameError,), _handler)