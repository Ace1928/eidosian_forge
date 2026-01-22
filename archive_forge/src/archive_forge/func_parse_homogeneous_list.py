from __future__ import absolute_import, division, print_function
from . import lexer, error
from . import coretypes
def parse_homogeneous_list(self, parse_item, sep_tok_id, errmsg, trailing_sep=False):
    """
        <item>_list : <item> <SEP> <item>_list
                    | <item>

        Returns a list of <item>s, or None.
        """
    saved_pos = self.pos
    items = []
    item = True
    while item is not None:
        item = parse_item()
        if item is not None:
            items.append(item)
            if self.tok.id == sep_tok_id:
                self.advance_tok()
            else:
                return items
        elif len(items) > 0:
            if trailing_sep:
                return items
            else:
                self.raise_error(errmsg)
        else:
            self.pos = saved_pos
            return None