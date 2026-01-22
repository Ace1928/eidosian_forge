import os as _os
import sys as _sys
import _thread
import functools
from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
from _thread import stack_size
def excepthook(args, /):
    """
        Handle uncaught Thread.run() exception.
        """
    if args.exc_type == SystemExit:
        return
    if _sys is not None and _sys.stderr is not None:
        stderr = _sys.stderr
    elif args.thread is not None:
        stderr = args.thread._stderr
        if stderr is None:
            return
    else:
        return
    if args.thread is not None:
        name = args.thread.name
    else:
        name = get_ident()
    print(f'Exception in thread {name}:', file=stderr, flush=True)
    _print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=stderr)
    stderr.flush()