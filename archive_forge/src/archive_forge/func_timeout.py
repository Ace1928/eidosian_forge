from typing import List, Optional, Union
def timeout(self, timeout: float) -> 'Query':
    """overrides the timeout parameter of the module"""
    self._timeout = timeout
    return self