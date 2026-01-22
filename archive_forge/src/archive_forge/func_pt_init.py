import asyncio
import os
import sys
from IPython.core.debugger import Pdb
from IPython.core.completer import IPCompleter
from .ptutils import IPythonPTCompleter
from .shortcuts import create_ipython_shortcuts
from . import embed
from pathlib import Path
from pygments.token import Token
from prompt_toolkit.application import create_app_session
from prompt_toolkit.shortcuts.prompt import PromptSession
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import PygmentsTokens
from prompt_toolkit.history import InMemoryHistory, FileHistory
from concurrent.futures import ThreadPoolExecutor
from prompt_toolkit import __version__ as ptk_version
def pt_init(self, pt_session_options=None):
    """Initialize the prompt session and the prompt loop
        and store them in self.pt_app and self.pt_loop.

        Additional keyword arguments for the PromptSession class
        can be specified in pt_session_options.
        """
    if pt_session_options is None:
        pt_session_options = {}

    def get_prompt_tokens():
        return [(Token.Prompt, self.prompt)]
    if self._ptcomp is None:
        compl = IPCompleter(shell=self.shell, namespace={}, global_namespace={}, parent=self.shell)
        methods_names = [m[3:] for m in dir(self) if m.startswith('do_')]

        def gen_comp(self, text):
            return [m for m in methods_names if m.startswith(text)]
        import types
        newcomp = types.MethodType(gen_comp, compl)
        compl.custom_matchers.insert(0, newcomp)
        self._ptcomp = IPythonPTCompleter(compl)
    if self.shell.debugger_history is None:
        if self.shell.debugger_history_file is not None:
            p = Path(self.shell.debugger_history_file).expanduser()
            if not p.exists():
                p.touch()
            self.debugger_history = FileHistory(os.path.expanduser(str(p)))
        else:
            self.debugger_history = InMemoryHistory()
    else:
        self.debugger_history = self.shell.debugger_history
    options = dict(message=lambda: PygmentsTokens(get_prompt_tokens()), editing_mode=getattr(EditingMode, self.shell.editing_mode.upper()), key_bindings=create_ipython_shortcuts(self.shell), history=self.debugger_history, completer=self._ptcomp, enable_history_search=True, mouse_support=self.shell.mouse_support, complete_style=self.shell.pt_complete_style, style=getattr(self.shell, 'style', None), color_depth=self.shell.color_depth)
    if not PTK3:
        options['inputhook'] = self.shell.inputhook
    options.update(pt_session_options)
    if not _use_simple_prompt:
        self.pt_loop = asyncio.new_event_loop()
        self.pt_app = PromptSession(**options)