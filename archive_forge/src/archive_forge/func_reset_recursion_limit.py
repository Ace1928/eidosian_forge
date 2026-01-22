import sys
def reset_recursion_limit(self):
    """Restores the recursion limit."""
    sys.setrecursionlimit(self.old_recursion_limit)