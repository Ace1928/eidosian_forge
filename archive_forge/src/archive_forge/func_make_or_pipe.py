import sys
import os
import socket
def make_or_pipe(pipe):
    """
    wraps a pipe into two pipe-like objects which are "or"d together to
    affect the real pipe. if either returned pipe is set, the wrapped pipe
    is set. when both are cleared, the wrapped pipe is cleared.
    """
    p1 = OrPipe(pipe)
    p2 = OrPipe(pipe)
    p1._partner = p2
    p2._partner = p1
    return (p1, p2)