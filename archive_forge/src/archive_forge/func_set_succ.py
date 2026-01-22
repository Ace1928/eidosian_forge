from typing import Dict, List, Optional
def set_succ(self, new_succ):
    """Set the node's successors."""
    if not isinstance(new_succ, type([])):
        raise NodeException('Node successor must be of list type.')
    self.succ = new_succ