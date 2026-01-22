from pygments.token import Token
import sys
from IPython.core.displayhook import DisplayHook
from prompt_toolkit.formatted_text import fragment_list_width, PygmentsTokens
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.enums import EditingMode
def vi_mode(self):
    if getattr(self.shell.pt_app, 'editing_mode', None) == EditingMode.VI and self.shell.prompt_includes_vi_mode:
        mode = str(self.shell.pt_app.app.vi_state.input_mode)
        if mode.startswith('InputMode.'):
            mode = mode[10:13].lower()
        elif mode.startswith('vi-'):
            mode = mode[3:6]
        return '[' + mode + '] '
    return ''