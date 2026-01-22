import contextlib
import os
import typing
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.sexp as sexp
from rpy2.robjects.robject import RObjectMixin
from rpy2.robjects import conversion
@contextlib.contextmanager
def local_context(env: typing.Optional[sexp.SexpEnvironment]=None, use_rlock: bool=True) -> typing.Iterator[Environment]:
    """Local context for the evaluation of R code.

    This is a wrapper around the rpy2.rinterface function with the
    same name.

    Args:
    - env: an environment to use as a context. If not specified (None, the
      default), a child environment to the current context is created.
    - use_rlock: whether to use a threading lock (see the documentation about
      "rlock". The default is True.

    Returns:
    Yield the environment (passed to env, or created).
    """
    with rinterface.local_context(env=env, use_rlock=use_rlock) as lc:
        yield Environment(lc)