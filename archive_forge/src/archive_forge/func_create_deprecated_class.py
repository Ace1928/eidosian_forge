import inspect
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, overload
from scrapy.exceptions import ScrapyDeprecationWarning
def create_deprecated_class(name: str, new_class: type, clsdict: Optional[Dict[str, Any]]=None, warn_category: Type[Warning]=ScrapyDeprecationWarning, warn_once: bool=True, old_class_path: Optional[str]=None, new_class_path: Optional[str]=None, subclass_warn_message: str='{cls} inherits from deprecated class {old}, please inherit from {new}.', instance_warn_message: str='{cls} is deprecated, instantiate {new} instead.') -> type:
    """
    Return a "deprecated" class that causes its subclasses to issue a warning.
    Subclasses of ``new_class`` are considered subclasses of this class.
    It also warns when the deprecated class is instantiated, but do not when
    its subclasses are instantiated.

    It can be used to rename a base class in a library. For example, if we
    have

        class OldName(SomeClass):
            # ...

    and we want to rename it to NewName, we can do the following::

        class NewName(SomeClass):
            # ...

        OldName = create_deprecated_class('OldName', NewName)

    Then, if user class inherits from OldName, warning is issued. Also, if
    some code uses ``issubclass(sub, OldName)`` or ``isinstance(sub(), OldName)``
    checks they'll still return True if sub is a subclass of NewName instead of
    OldName.
    """

    class DeprecatedClass(new_class.__class__):
        deprecated_class: Optional[type] = None
        warned_on_subclass: bool = False

        def __new__(metacls, name: str, bases: Tuple[type, ...], clsdict_: Dict[str, Any]) -> type:
            cls = super().__new__(metacls, name, bases, clsdict_)
            if metacls.deprecated_class is None:
                metacls.deprecated_class = cls
            return cls

        def __init__(cls, name: str, bases: Tuple[type, ...], clsdict_: Dict[str, Any]):
            meta = cls.__class__
            old = meta.deprecated_class
            if old in bases and (not (warn_once and meta.warned_on_subclass)):
                meta.warned_on_subclass = True
                msg = subclass_warn_message.format(cls=_clspath(cls), old=_clspath(old, old_class_path), new=_clspath(new_class, new_class_path))
                if warn_once:
                    msg += ' (warning only on first subclass, there may be others)'
                warnings.warn(msg, warn_category, stacklevel=2)
            super().__init__(name, bases, clsdict_)

        def __instancecheck__(cls, inst: Any) -> bool:
            return any((cls.__subclasscheck__(c) for c in (type(inst), inst.__class__)))

        def __subclasscheck__(cls, sub: type) -> bool:
            if cls is not DeprecatedClass.deprecated_class:
                return super().__subclasscheck__(sub)
            if not inspect.isclass(sub):
                raise TypeError('issubclass() arg 1 must be a class')
            mro = getattr(sub, '__mro__', ())
            return any((c in {cls, new_class} for c in mro))

        def __call__(cls, *args: Any, **kwargs: Any) -> Any:
            old = DeprecatedClass.deprecated_class
            if cls is old:
                msg = instance_warn_message.format(cls=_clspath(cls, old_class_path), new=_clspath(new_class, new_class_path))
                warnings.warn(msg, warn_category, stacklevel=2)
            return super().__call__(*args, **kwargs)
    deprecated_cls = DeprecatedClass(name, (new_class,), clsdict or {})
    try:
        frm = inspect.stack()[1]
        parent_module = inspect.getmodule(frm[0])
        if parent_module is not None:
            deprecated_cls.__module__ = parent_module.__name__
    except Exception as e:
        warnings.warn(f'Error detecting parent module: {e!r}')
    return deprecated_cls