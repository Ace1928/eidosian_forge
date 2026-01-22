from typing import Any, Dict, Iterator, List, Set, no_type_check
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from qpd.constants import AGGREGATION_FUNCTIONS, WINDOW_FUNCTIONS
@property
def has_windowframe(self) -> bool:
    return not isinstance(self.window.windowframe, NoWindowFrame)