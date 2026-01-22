from __future__ import annotations
import multiprocessing
import multiprocessing.process
import os
import os.path
import sys
import traceback
from typing import Any
from coverage.debug import DebugControl
def patch_multiprocessing(rcfile: str) -> None:
    """Monkey-patch the multiprocessing module.

    This enables coverage measurement of processes started by multiprocessing.
    This involves aggressive monkey-patching.

    `rcfile` is the path to the rcfile being used.

    """
    if hasattr(multiprocessing, PATCHED_MARKER):
        return
    OriginalProcess._bootstrap = ProcessWithCoverage._bootstrap
    os.environ['COVERAGE_RCFILE'] = os.path.abspath(rcfile)
    try:
        from multiprocessing import spawn
        original_get_preparation_data = spawn.get_preparation_data
    except (ImportError, AttributeError):
        pass
    else:

        def get_preparation_data_with_stowaway(name: str) -> dict[str, Any]:
            """Get the original preparation data, and also insert our stowaway."""
            d = original_get_preparation_data(name)
            d['stowaway'] = Stowaway(rcfile)
            return d
        spawn.get_preparation_data = get_preparation_data_with_stowaway
    setattr(multiprocessing, PATCHED_MARKER, True)