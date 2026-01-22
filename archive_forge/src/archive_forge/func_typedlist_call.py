from collections.abc import MutableSequence
from numba.core.types import ListType
from numba.core.imputils import numba_typeref_ctor
from numba.core.dispatcher import Dispatcher
from numba.core import types, config, cgutils
from numba import njit, typeof
from numba.core.extending import (
from numba.typed import listobject
from numba.core.errors import TypingError, LoweringError
from numba.core.typing.templates import Signature
import typing as pt
@type_callable(ListType)
def typedlist_call(context):
    """Defines typing logic for ``List()`` and ``List(iterable)``.

    If no argument is given, the returned typer types a new typed-list with an
    undefined item type. If a single argument is given it must be iterable with
    a guessable 'dtype'. In this case, the typer types a new typed-list with
    the type set to the 'dtype' of the iterable arg.

    Parameters
    ----------
    arg : single iterable (optional)
        The single optional argument.

    Returns
    -------
    typer : function
        A typer suitable to type constructor calls.

    Raises
    ------
    The returned typer raises a TypingError in case of unsuitable arguments.

    """

    class Typer(object):

        def attach_sig(self):
            from inspect import signature as mypysig

            def mytyper(iterable):
                pass
            self.pysig = mypysig(mytyper)

        def __call__(self, *args, **kwargs):
            if kwargs:
                raise TypingError('List() takes no keyword arguments')
            elif args:
                if not 0 <= len(args) <= 1:
                    raise TypingError('List() expected at most 1 argument, got {}'.format(len(args)))
                rt = types.ListType(_guess_dtype(args[0]))
                self.attach_sig()
                return Signature(rt, args, None, pysig=self.pysig)
            else:
                item_type = types.undefined
                return types.ListType(item_type)
    return Typer()