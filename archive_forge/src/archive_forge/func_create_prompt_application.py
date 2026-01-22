from __future__ import unicode_literals
from .buffer import Buffer, AcceptAction
from .document import Document
from .enums import DEFAULT_BUFFER, SEARCH_BUFFER, EditingMode
from .filters import IsDone, HasFocus, RendererHeightIsKnown, to_simple_filter, to_cli_filter, Condition
from .history import InMemoryHistory
from .interface import CommandLineInterface, Application, AbortAction
from .key_binding.defaults import load_key_bindings_for_prompt
from .key_binding.registry import Registry
from .keys import Keys
from .layout import Window, HSplit, FloatContainer, Float
from .layout.containers import ConditionalContainer
from .layout.controls import BufferControl, TokenListControl
from .layout.dimension import LayoutDimension
from .layout.lexers import PygmentsLexer
from .layout.margins import PromptMargin, ConditionalMargin
from .layout.menus import CompletionsMenu, MultiColumnCompletionsMenu
from .layout.processors import PasswordProcessor, ConditionalProcessor, AppendAutoSuggestion, HighlightSearchProcessor, HighlightSelectionProcessor, DisplayMultipleCursors
from .layout.prompt import DefaultPrompt
from .layout.screen import Char
from .layout.toolbars import ValidationToolbar, SystemToolbar, ArgToolbar, SearchToolbar
from .layout.utils import explode_tokens
from .renderer import print_tokens as renderer_print_tokens
from .styles import DEFAULT_STYLE, Style, style_from_dict
from .token import Token
from .utils import is_conemu_ansi, is_windows, DummyContext
from six import text_type, exec_, PY2
import os
import sys
import textwrap
import threading
import time
def create_prompt_application(message='', multiline=False, wrap_lines=True, is_password=False, vi_mode=False, editing_mode=EditingMode.EMACS, complete_while_typing=True, enable_history_search=False, lexer=None, enable_system_bindings=False, enable_open_in_editor=False, validator=None, completer=None, reserve_space_for_menu=8, auto_suggest=None, style=None, history=None, clipboard=None, get_prompt_tokens=None, get_continuation_tokens=None, get_rprompt_tokens=None, get_bottom_toolbar_tokens=None, display_completions_in_columns=False, get_title=None, mouse_support=False, extra_input_processors=None, key_bindings_registry=None, on_abort=AbortAction.RAISE_EXCEPTION, on_exit=AbortAction.RAISE_EXCEPTION, accept_action=AcceptAction.RETURN_DOCUMENT, erase_when_done=False, default=''):
    """
    Create an :class:`~Application` instance for a prompt.

    (It is meant to cover 90% of the prompt use cases, where no extreme
    customization is required. For more complex input, it is required to create
    a custom :class:`~Application` instance.)

    :param message: Text to be shown before the prompt.
    :param mulitiline: Allow multiline input. Pressing enter will insert a
                       newline. (This requires Meta+Enter to accept the input.)
    :param wrap_lines: `bool` or :class:`~prompt_toolkit.filters.CLIFilter`.
        When True (the default), automatically wrap long lines instead of
        scrolling horizontally.
    :param is_password: Show asterisks instead of the actual typed characters.
    :param editing_mode: ``EditingMode.VI`` or ``EditingMode.EMACS``.
    :param vi_mode: `bool`, if True, Identical to ``editing_mode=EditingMode.VI``.
    :param complete_while_typing: `bool` or
        :class:`~prompt_toolkit.filters.SimpleFilter`. Enable autocompletion
        while typing.
    :param enable_history_search: `bool` or
        :class:`~prompt_toolkit.filters.SimpleFilter`. Enable up-arrow parting
        string matching.
    :param lexer: :class:`~prompt_toolkit.layout.lexers.Lexer` to be used for
        the syntax highlighting.
    :param validator: :class:`~prompt_toolkit.validation.Validator` instance
        for input validation.
    :param completer: :class:`~prompt_toolkit.completion.Completer` instance
        for input completion.
    :param reserve_space_for_menu: Space to be reserved for displaying the menu.
        (0 means that no space needs to be reserved.)
    :param auto_suggest: :class:`~prompt_toolkit.auto_suggest.AutoSuggest`
        instance for input suggestions.
    :param style: :class:`.Style` instance for the color scheme.
    :param enable_system_bindings: `bool` or
        :class:`~prompt_toolkit.filters.CLIFilter`. Pressing Meta+'!' will show
        a system prompt.
    :param enable_open_in_editor: `bool` or
        :class:`~prompt_toolkit.filters.CLIFilter`. Pressing 'v' in Vi mode or
        C-X C-E in emacs mode will open an external editor.
    :param history: :class:`~prompt_toolkit.history.History` instance.
    :param clipboard: :class:`~prompt_toolkit.clipboard.base.Clipboard` instance.
        (e.g. :class:`~prompt_toolkit.clipboard.in_memory.InMemoryClipboard`)
    :param get_bottom_toolbar_tokens: Optional callable which takes a
        :class:`~prompt_toolkit.interface.CommandLineInterface` and returns a
        list of tokens for the bottom toolbar.
    :param display_completions_in_columns: `bool` or
        :class:`~prompt_toolkit.filters.CLIFilter`. Display the completions in
        multiple columns.
    :param get_title: Callable that returns the title to be displayed in the
        terminal.
    :param mouse_support: `bool` or :class:`~prompt_toolkit.filters.CLIFilter`
        to enable mouse support.
    :param default: The default text to be shown in the input buffer. (This can
        be edited by the user.)
    """
    if key_bindings_registry is None:
        key_bindings_registry = load_key_bindings_for_prompt(enable_system_bindings=enable_system_bindings, enable_open_in_editor=enable_open_in_editor)
    if vi_mode:
        editing_mode = EditingMode.VI
    complete_while_typing = to_simple_filter(complete_while_typing)
    enable_history_search = to_simple_filter(enable_history_search)
    multiline = to_simple_filter(multiline)
    complete_while_typing = complete_while_typing & ~enable_history_search
    try:
        if pygments_Style and issubclass(style, pygments_Style):
            style = style_from_dict(style.styles)
    except TypeError:
        pass
    return Application(layout=create_prompt_layout(message=message, lexer=lexer, is_password=is_password, reserve_space_for_menu=reserve_space_for_menu if completer is not None else 0, multiline=Condition(lambda cli: multiline()), get_prompt_tokens=get_prompt_tokens, get_continuation_tokens=get_continuation_tokens, get_rprompt_tokens=get_rprompt_tokens, get_bottom_toolbar_tokens=get_bottom_toolbar_tokens, display_completions_in_columns=display_completions_in_columns, extra_input_processors=extra_input_processors, wrap_lines=wrap_lines), buffer=Buffer(enable_history_search=enable_history_search, complete_while_typing=complete_while_typing, is_multiline=multiline, history=history or InMemoryHistory(), validator=validator, completer=completer, auto_suggest=auto_suggest, accept_action=accept_action, initial_document=Document(default)), style=style or DEFAULT_STYLE, clipboard=clipboard, key_bindings_registry=key_bindings_registry, get_title=get_title, mouse_support=mouse_support, editing_mode=editing_mode, erase_when_done=erase_when_done, reverse_vi_search_direction=True, on_abort=on_abort, on_exit=on_exit)