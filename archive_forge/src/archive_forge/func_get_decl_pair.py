from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
def get_decl_pair(self):
    """ See Declarator.get_decl_pair."""

    def get_tp():
        """ Iterator generating lines for struct definition. """
        decl = 'struct '
        if self.tpname is not None:
            decl += self.tpname
            if self.inherit is not None:
                decl += ' : ' + self.inherit
        yield decl
        yield '{'
        for f in self.fields:
            for f_line in f.generate():
                yield ('  ' + f_line)
        yield '} '
    return (get_tp(), '')