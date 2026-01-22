import re
from typing import Dict, Any
def render_tokens(self, tokens, state):
    return ''.join(self.iter_tokens(tokens, state))