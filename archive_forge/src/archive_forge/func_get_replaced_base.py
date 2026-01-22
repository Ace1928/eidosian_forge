import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def get_replaced_base(self, variables):
    if self._may_have_internal_variables:
        return variables.replace_string(self.base)
    return self.base