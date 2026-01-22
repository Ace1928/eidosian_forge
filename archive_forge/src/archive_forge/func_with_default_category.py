from typing import (
from .constants import (
from .exceptions import (
from .utils import (
def with_default_category(category: str, *, heritable: bool=True) -> Callable[[Type['CommandSet']], Type['CommandSet']]:
    """
    Decorator that applies a category to all ``do_*`` command methods in a class that do not already
    have a category specified.

    CommandSets that are decorated by this with `heritable` set to True (default) will set a class attribute that is
    inherited by all subclasses unless overridden. All commands of this CommandSet and all subclasses of this CommandSet
    that do not declare an explicit category will be placed in this category. Subclasses may use this decorator to
    override the default category.

    If `heritable` is set to False, then only the commands declared locally to this CommandSet will be placed in the
    specified category. Dynamically created commands, and commands declared in sub-classes will not receive this
    category.

    :param category: category to put all uncategorized commands in
    :param heritable: Flag whether this default category should apply to sub-classes. Defaults to True
    :return: decorator function
    """

    def decorate_class(cls: Type[CommandSet]) -> Type[CommandSet]:
        if heritable:
            setattr(cls, CLASS_ATTR_DEFAULT_HELP_CATEGORY, category)
        import inspect
        from .constants import CMD_ATTR_HELP_CATEGORY
        from .decorators import with_category
        methods = inspect.getmembers(cls, predicate=lambda meth: inspect.isfunction(meth) and meth.__name__.startswith(COMMAND_FUNC_PREFIX) and (meth in inspect.getmro(cls)[0].__dict__.values()))
        category_decorator = with_category(category)
        for method in methods:
            if not hasattr(method[1], CMD_ATTR_HELP_CATEGORY):
                setattr(cls, method[0], category_decorator(method[1]))
        return cls
    return decorate_class