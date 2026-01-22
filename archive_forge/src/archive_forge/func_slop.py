from typing import List, Optional, Union
def slop(self, slop: int) -> 'Query':
    """Allow a maximum of N intervening non matched terms between
        phrase terms (0 means exact phrase).
        """
    self._slop = slop
    return self