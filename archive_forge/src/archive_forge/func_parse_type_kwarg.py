from __future__ import absolute_import, division, print_function
from . import lexer, error
from . import coretypes
def parse_type_kwarg(self):
    """
        type_kwarg : NAME_LOWER EQUAL type_arg

        Returns a (name, type_arg) tuple, or None.
        """
    if self.tok.id != lexer.NAME_LOWER:
        return None
    saved_pos = self.pos
    name = self.tok.val
    self.advance_tok()
    if self.tok.id != lexer.EQUAL:
        self.pos = saved_pos
        return None
    self.advance_tok()
    arg = self.parse_type_arg()
    if arg is not None:
        return (name, arg)
    else:
        self.raise_error('Expected a type constructor argument')