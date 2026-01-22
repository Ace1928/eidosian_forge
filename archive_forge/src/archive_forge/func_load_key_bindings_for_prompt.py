from __future__ import unicode_literals
from prompt_toolkit.key_binding.registry import ConditionalRegistry, MergedRegistry
from prompt_toolkit.key_binding.bindings.basic import load_basic_bindings, load_abort_and_exit_bindings, load_basic_system_bindings, load_auto_suggestion_bindings, load_mouse_bindings
from prompt_toolkit.key_binding.bindings.emacs import load_emacs_bindings, load_emacs_system_bindings, load_emacs_search_bindings, load_emacs_open_in_editor_bindings, load_extra_emacs_page_navigation_bindings
from prompt_toolkit.key_binding.bindings.vi import load_vi_bindings, load_vi_system_bindings, load_vi_search_bindings, load_vi_open_in_editor_bindings, load_extra_vi_page_navigation_bindings
from prompt_toolkit.filters import to_cli_filter
def load_key_bindings_for_prompt(**kw):
    """
    Create a ``Registry`` object with the defaults key bindings for an input
    prompt.

    This activates the key bindings for abort/exit (Ctrl-C/Ctrl-D),
    incremental search and auto suggestions.

    (Not for full screen applications.)
    """
    kw.setdefault('enable_abort_and_exit_bindings', True)
    kw.setdefault('enable_search', True)
    kw.setdefault('enable_auto_suggest_bindings', True)
    return load_key_bindings(**kw)