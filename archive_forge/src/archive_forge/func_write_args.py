from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def write_args(args, defaults):
    no_default_count = len(args) - len(defaults)
    for i, arg in enumerate(args):
        write_possible_comma()
        self.visit(arg)
        default_idx = i - no_default_count
        if default_idx >= 0 and defaults[default_idx] is not None:
            self._write('=')
            self.visit(defaults[i - no_default_count])