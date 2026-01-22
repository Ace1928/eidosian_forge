import bz2
import collections
import gzip
import inspect
import itertools
import re
import warnings
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from inspect import Parameter, signature
from os.path import splitext
from pathlib import Path
import networkx as nx
from networkx.utils import create_py_random_state, create_random_state
def py_random_state(random_state_argument):
    """Decorator to generate a random.Random instance (or equiv).

    The decorator processes the argument indicated by `random_state_argument`
    using :func:`nx.utils.create_py_random_state`.
    The argument value can be a seed (integer), or a random number generator::

        If int, return a random.Random instance set with seed=int.
        If random.Random instance, return it.
        If None or the `random` package, return the global random number
        generator used by `random`.
        If np.random package, return the global numpy random number
        generator wrapped in a PythonRandomInterface class.
        If np.random.RandomState instance, return it wrapped in
        PythonRandomInterface
        If a PythonRandomInterface instance, return it

    Parameters
    ----------
    random_state_argument : string or int
        The name of the argument or the index of the argument in args that is
        to be converted to the random.Random instance or numpy.random.RandomState
        instance that mimics basic methods of random.Random.

    Returns
    -------
    _random_state : function
        Function whose random_state_argument is converted to a Random instance.

    Examples
    --------
    Decorate functions like this::

       @py_random_state("random_state")
       def random_float(random_state=None):
           return random_state.rand()

       @py_random_state(0)
       def random_float(rng=None):
           return rng.rand()

       @py_random_state(1)
       def random_array(dims, seed=12345):
           return seed.rand(*dims)

    See Also
    --------
    np_random_state
    """
    return argmap(create_py_random_state, random_state_argument)