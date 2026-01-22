import logging
def maxIgnoreNone(a, b):
    """
    Return the max of two numbers, ignoring None
    """
    if a is None:
        return b
    if b is None:
        return a
    if a < b:
        return b
    return a