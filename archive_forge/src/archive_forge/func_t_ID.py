import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(identifier)
def t_ID(self, t):
    t.type = self.keyword_map.get(t.value, 'ID')
    if t.type == 'ID' and self.type_lookup_func(t.value):
        t.type = 'TYPEID'
    return t