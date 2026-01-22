from __future__ import annotations
import contextlib
import datetime
from functools import partial
from functools import wraps
import json
import logging
from numbers import Number
import threading
import time
from typing import Any
from typing import Callable
from typing import cast
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from decorator import decorate
from . import exception
from .api import BackendArguments
from .api import BackendFormatted
from .api import CachedValue
from .api import CacheMutex
from .api import CacheReturnType
from .api import CantDeserializeException
from .api import KeyType
from .api import MetaDataType
from .api import NO_VALUE
from .api import SerializedReturnType
from .api import Serializer
from .api import ValuePayload
from .backends import _backend_loader
from .backends import register_backend  # noqa
from .proxy import ProxyBackend
from .util import function_key_generator
from .util import function_multi_key_generator
from .util import repr_obj
from .. import Lock
from .. import NeedRegenerationException
from ..util import coerce_string_conf
from ..util import memoized_property
from ..util import NameRegistry
from ..util import PluginLoader
from ..util.typing import Self
def cache_on_arguments(self, namespace: Optional[str]=None, expiration_time: Union[float, ExpirationTimeCallable, None]=None, should_cache_fn: Optional[Callable[[ValuePayload], bool]]=None, to_str: Callable[[Any], str]=str, function_key_generator: Optional[FunctionKeyGenerator]=None) -> Callable[[Callable[..., ValuePayload]], Callable[..., ValuePayload]]:
    """A function decorator that will cache the return
        value of the function using a key derived from the
        function itself and its arguments.

        The decorator internally makes use of the
        :meth:`.CacheRegion.get_or_create` method to access the
        cache and conditionally call the function.  See that
        method for additional behavioral details.

        E.g.::

            @someregion.cache_on_arguments()
            def generate_something(x, y):
                return somedatabase.query(x, y)

        The decorated function can then be called normally, where
        data will be pulled from the cache region unless a new
        value is needed::

            result = generate_something(5, 6)

        The function is also given an attribute ``invalidate()``, which
        provides for invalidation of the value.  Pass to ``invalidate()``
        the same arguments you'd pass to the function itself to represent
        a particular value::

            generate_something.invalidate(5, 6)

        Another attribute ``set()`` is added to provide extra caching
        possibilities relative to the function.   This is a convenience
        method for :meth:`.CacheRegion.set` which will store a given
        value directly without calling the decorated function.
        The value to be cached is passed as the first argument, and the
        arguments which would normally be passed to the function
        should follow::

            generate_something.set(3, 5, 6)

        The above example is equivalent to calling
        ``generate_something(5, 6)``, if the function were to produce
        the value ``3`` as the value to be cached.

        .. versionadded:: 0.4.1 Added ``set()`` method to decorated function.

        Similar to ``set()`` is ``refresh()``.   This attribute will
        invoke the decorated function and populate a new value into
        the cache with the new value, as well as returning that value::

            newvalue = generate_something.refresh(5, 6)

        .. versionadded:: 0.5.0 Added ``refresh()`` method to decorated
           function.

        ``original()`` on other hand will invoke the decorated function
        without any caching::

            newvalue = generate_something.original(5, 6)

        .. versionadded:: 0.6.0 Added ``original()`` method to decorated
           function.

        Lastly, the ``get()`` method returns either the value cached
        for the given key, or the token ``NO_VALUE`` if no such key
        exists::

            value = generate_something.get(5, 6)

        .. versionadded:: 0.5.3 Added ``get()`` method to decorated
           function.

        The default key generation will use the name
        of the function, the module name for the function,
        the arguments passed, as well as an optional "namespace"
        parameter in order to generate a cache key.

        Given a function ``one`` inside the module
        ``myapp.tools``::

            @region.cache_on_arguments(namespace="foo")
            def one(a, b):
                return a + b

        Above, calling ``one(3, 4)`` will produce a
        cache key as follows::

            myapp.tools:one|foo|3 4

        The key generator will ignore an initial argument
        of ``self`` or ``cls``, making the decorator suitable
        (with caveats) for use with instance or class methods.
        Given the example::

            class MyClass:
                @region.cache_on_arguments(namespace="foo")
                def one(self, a, b):
                    return a + b

        The cache key above for ``MyClass().one(3, 4)`` will
        again produce the same cache key of ``myapp.tools:one|foo|3 4`` -
        the name ``self`` is skipped.

        The ``namespace`` parameter is optional, and is used
        normally to disambiguate two functions of the same
        name within the same module, as can occur when decorating
        instance or class methods as below::

            class MyClass:
                @region.cache_on_arguments(namespace='MC')
                def somemethod(self, x, y):
                    ""

            class MyOtherClass:
                @region.cache_on_arguments(namespace='MOC')
                def somemethod(self, x, y):
                    ""

        Above, the ``namespace`` parameter disambiguates
        between ``somemethod`` on ``MyClass`` and ``MyOtherClass``.
        Python class declaration mechanics otherwise prevent
        the decorator from having awareness of the ``MyClass``
        and ``MyOtherClass`` names, as the function is received
        by the decorator before it becomes an instance method.

        The function key generation can be entirely replaced
        on a per-region basis using the ``function_key_generator``
        argument present on :func:`.make_region` and
        :class:`.CacheRegion`. If defaults to
        :func:`.function_key_generator`.

        :param namespace: optional string argument which will be
         established as part of the cache key.   This may be needed
         to disambiguate functions of the same name within the same
         source file, such as those
         associated with classes - note that the decorator itself
         can't see the parent class on a function as the class is
         being declared.

        :param expiration_time: if not None, will override the normal
         expiration time.

         May be specified as a callable, taking no arguments, that
         returns a value to be used as the ``expiration_time``. This callable
         will be called whenever the decorated function itself is called, in
         caching or retrieving. Thus, this can be used to
         determine a *dynamic* expiration time for the cached function
         result.  Example use cases include "cache the result until the
         end of the day, week or time period" and "cache until a certain date
         or time passes".

        :param should_cache_fn: passed to :meth:`.CacheRegion.get_or_create`.

        :param to_str: callable, will be called on each function argument
         in order to convert to a string.  Defaults to ``str()``.  If the
         function accepts non-ascii unicode arguments on Python 2.x, the
         ``unicode()`` builtin can be substituted, but note this will
         produce unicode cache keys which may require key mangling before
         reaching the cache.

        :param function_key_generator: a function that will produce a
         "cache key". This function will supersede the one configured on the
         :class:`.CacheRegion` itself.

        .. seealso::

            :meth:`.CacheRegion.cache_multi_on_arguments`

            :meth:`.CacheRegion.get_or_create`

        """
    expiration_time_is_callable = callable(expiration_time)
    if function_key_generator is None:
        _function_key_generator = self.function_key_generator
    else:
        _function_key_generator = function_key_generator

    def get_or_create_for_user_func(key_generator, user_func, *arg, **kw):
        key = key_generator(*arg, **kw)
        timeout: Optional[float] = cast(ExpirationTimeCallable, expiration_time)() if expiration_time_is_callable else cast(Optional[float], expiration_time)
        return self.get_or_create(key, user_func, timeout, should_cache_fn, (arg, kw))

    def cache_decorator(user_func):
        if to_str is cast(Callable[[Any], str], str):
            key_generator = _function_key_generator(namespace, user_func)
        else:
            key_generator = _function_key_generator(namespace, user_func, to_str)

        def refresh(*arg, **kw):
            """
                Like invalidate, but regenerates the value instead
                """
            key = key_generator(*arg, **kw)
            value = user_func(*arg, **kw)
            self.set(key, value)
            return value

        def invalidate(*arg, **kw):
            key = key_generator(*arg, **kw)
            self.delete(key)

        def set_(value, *arg, **kw):
            key = key_generator(*arg, **kw)
            self.set(key, value)

        def get(*arg, **kw):
            key = key_generator(*arg, **kw)
            return self.get(key)
        user_func.set = set_
        user_func.invalidate = invalidate
        user_func.get = get
        user_func.refresh = refresh
        user_func.original = user_func
        return decorate(user_func, partial(get_or_create_for_user_func, key_generator))
    return cache_decorator