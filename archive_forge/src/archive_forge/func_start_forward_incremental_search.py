from __future__ import annotations
from prompt_toolkit import search
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition, control_is_searchable, is_searching
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from ..key_bindings import key_binding
@key_binding(filter=control_is_searchable)
def start_forward_incremental_search(event: E) -> None:
    """
    Enter forward incremental search.
    (Usually ControlS.)
    """
    search.start_search(direction=search.SearchDirection.FORWARD)