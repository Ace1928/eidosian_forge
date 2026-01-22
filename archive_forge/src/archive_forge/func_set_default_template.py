import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def set_default_template(self, template):
    self._default_template = self._is_template_set(template)