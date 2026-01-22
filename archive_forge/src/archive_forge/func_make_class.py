import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
def make_class(name, attrs, bases=(object,), class_body=None, **attributes_arguments):
    """
    A quick way to create a new class called *name* with *attrs*.

    :param str name: The name for the new class.

    :param attrs: A list of names or a dictionary of mappings of names to
        `attr.ib`\\ s / `attrs.field`\\ s.

        The order is deduced from the order of the names or attributes inside
        *attrs*.  Otherwise the order of the definition of the attributes is
        used.
    :type attrs: `list` or `dict`

    :param tuple bases: Classes that the new class will subclass.

    :param dict class_body: An optional dictionary of class attributes for the new class.

    :param attributes_arguments: Passed unmodified to `attr.s`.

    :return: A new class with *attrs*.
    :rtype: type

    .. versionadded:: 17.1.0 *bases*
    .. versionchanged:: 18.1.0 If *attrs* is ordered, the order is retained.
    .. versionchanged:: 23.2.0 *class_body*
    """
    if isinstance(attrs, dict):
        cls_dict = attrs
    elif isinstance(attrs, (list, tuple)):
        cls_dict = {a: attrib() for a in attrs}
    else:
        msg = 'attrs argument must be a dict or a list.'
        raise TypeError(msg)
    pre_init = cls_dict.pop('__attrs_pre_init__', None)
    post_init = cls_dict.pop('__attrs_post_init__', None)
    user_init = cls_dict.pop('__init__', None)
    body = {}
    if class_body is not None:
        body.update(class_body)
    if pre_init is not None:
        body['__attrs_pre_init__'] = pre_init
    if post_init is not None:
        body['__attrs_post_init__'] = post_init
    if user_init is not None:
        body['__init__'] = user_init
    type_ = types.new_class(name, bases, {}, lambda ns: ns.update(body))
    with contextlib.suppress(AttributeError, ValueError):
        type_.__module__ = sys._getframe(1).f_globals.get('__name__', '__main__')
    cmp = attributes_arguments.pop('cmp', None)
    attributes_arguments['eq'], attributes_arguments['order'] = _determine_attrs_eq_order(cmp, attributes_arguments.get('eq'), attributes_arguments.get('order'), True)
    return _attrs(these=cls_dict, **attributes_arguments)(type_)