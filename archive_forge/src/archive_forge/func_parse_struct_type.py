from __future__ import absolute_import, division, print_function
from . import lexer, error
from . import coretypes
def parse_struct_type(self):
    """
        struct_type : LBRACE struct_field_list RBRACE
                    | LBRACE struct_field_list COMMA RBRACE

        Returns a struct type, or None.
        """
    if self.tok.id != lexer.LBRACE:
        return None
    saved_pos = self.pos
    self.advance_tok()
    fields = self.parse_homogeneous_list(self.parse_struct_field, lexer.COMMA, 'Invalid field in struct', trailing_sep=True) or []
    if self.tok.id != lexer.RBRACE:
        self.raise_error('Invalid field in struct')
    self.advance_tok()
    names = [f[0] for f in fields]
    types = [f[1] for f in fields]
    tconstr = self.syntactic_sugar(self.sym.dtype_constr, 'struct', '{...} dtype constructor', saved_pos)
    return tconstr(names, types)