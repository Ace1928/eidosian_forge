from __future__ import absolute_import, division, print_function
from . import lexer, error
from . import coretypes
def parse_datashape(self):
    """
        datashape : datashape_nooption
                  | QUESTIONMARK datashape_nooption
                  | EXCLAMATIONMARK datashape_nooption

        Returns a datashape object or None.
        """
    tok = self.tok
    constructors = {lexer.QUESTIONMARK: 'option'}
    if tok.id in constructors:
        self.advance_tok()
        saved_pos = self.pos
        ds = self.parse_datashape_nooption()
        if ds is not None:
            option = self.syntactic_sugar(self.sym.dtype_constr, constructors[tok.id], '%s dtype construction' % constructors[tok.id], saved_pos - 1)
            return coretypes.DataShape(option(ds))
    else:
        return self.parse_datashape_nooption()