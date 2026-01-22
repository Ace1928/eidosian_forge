import codecs
import re
from yaql.language import exceptions
def t_KEYWORD_STRING(self, t):
    """
        (?!__)\\b[^\\W\\d]\\w*\\b
        """
    if t.value in self._operators_table:
        t.type = self._operators_table[t.value][2]
    else:
        t.type = self.keywords.get(t.value, 'KEYWORD_STRING')
        t.value = self.keyword_to_val.get(t.type, t.value)
    return t