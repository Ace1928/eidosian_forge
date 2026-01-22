from pygments.token import Token
import sys
from IPython.core.displayhook import DisplayHook
from prompt_toolkit.formatted_text import fragment_list_width, PygmentsTokens
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.enums import EditingMode
def in_prompt_tokens(self):
    return [(Token.Prompt, '>>> ')]