import io
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from cProfile import Profile
from functools import wraps
from ..config import config
from ..util import escape
from .state import state
@contextmanager
def profile_ctx(engine='pyinstrument'):
    """
    A context manager which profiles the body of the with statement
    with the supplied profiling engine and returns the profiling object
    in a list.

    Arguments
    ---------
    engine: str
      The profiling engine, e.g. 'pyinstrument', 'snakeviz' or 'memray'

    Returns
    -------
    sessions: list
      A list containing the profiling session.
    """
    if engine == 'pyinstrument':
        from pyinstrument import Profiler
        try:
            prof = Profiler()
            prof.start()
        except RuntimeError:
            prof = Profiler(async_mode='disabled')
            prof.start()
    elif engine == 'snakeviz':
        prof = Profile()
        prof.enable()
    elif engine == 'memray':
        import memray
        tmp_file = f'{tempfile.gettempdir()}/tmp{uuid.uuid4().hex}'
        tracker = memray.Tracker(tmp_file)
        tracker.__enter__()
    elif engine is None:
        pass
    sessions = []
    yield sessions
    if engine == 'pyinstrument':
        sessions.append(prof.stop())
    elif engine == 'snakeviz':
        prof.disable()
        sessions.append(prof)
    elif engine == 'memray':
        tracker.__exit__(None, None, None)
        sessions.append(open(tmp_file, 'rb').read())
        os.remove(tmp_file)