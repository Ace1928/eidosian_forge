from __future__ import annotations
from collections import deque
from time import monotonic
def expected_time(self, tokens=1):
    """Return estimated time of token availability.

        Returns
        -------
            float: the time in seconds.
        """
    _tokens = self._get_tokens()
    tokens = max(tokens, _tokens)
    return (tokens - _tokens) / self.fill_rate