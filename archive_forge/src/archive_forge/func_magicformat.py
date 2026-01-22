import sys
def magicformat(format):
    """Evaluate and substitute the appropriate parts of the string."""
    frame = sys._getframe(1)
    return dictformat(format, frame.f_locals, frame.f_globals)