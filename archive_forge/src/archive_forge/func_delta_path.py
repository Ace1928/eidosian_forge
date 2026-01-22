from __future__ import annotations
from typing import Any
from streamlit import util
from streamlit.runtime.scriptrunner import get_script_run_ctx
@property
def delta_path(self) -> list[int]:
    """The complete path of the delta pointed to by this cursor - its
        container, parent path, and index.
        """
    return make_delta_path(self.root_container, self.parent_path, self.index)