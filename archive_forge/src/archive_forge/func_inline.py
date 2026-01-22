from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
def inline(self):
    """Return the declarator as a single line."""
    tp_lines, tp_decl = self.get_decl_pair()
    tp_lines = ' '.join(tp_lines)
    if tp_decl is None:
        return tp_lines
    else:
        return '%s %s' % (tp_lines, tp_decl)