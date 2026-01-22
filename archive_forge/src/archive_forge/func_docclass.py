import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def docclass(self, object, name=None, mod=None, *ignored):
    """Produce text documentation for a given class object."""
    realname = object.__name__
    name = name or realname
    bases = object.__bases__

    def makename(c, m=object.__module__):
        return classname(c, m)
    if name == realname:
        title = 'class ' + self.bold(realname)
    else:
        title = self.bold(name) + ' = class ' + realname
    if bases:
        parents = map(makename, bases)
        title = title + '(%s)' % ', '.join(parents)
    contents = []
    push = contents.append
    try:
        signature = inspect.signature(object)
    except (ValueError, TypeError):
        signature = None
    if signature:
        argspec = str(signature)
        if argspec and argspec != '()':
            push(name + argspec + '\n')
    doc = getdoc(object)
    if doc:
        push(doc + '\n')
    mro = deque(inspect.getmro(object))
    if len(mro) > 2:
        push('Method resolution order:')
        for base in mro:
            push('    ' + makename(base))
        push('')
    subclasses = sorted((str(cls.__name__) for cls in type.__subclasses__(object) if not cls.__name__.startswith('_') and cls.__module__ == 'builtins'), key=str.lower)
    no_of_subclasses = len(subclasses)
    MAX_SUBCLASSES_TO_DISPLAY = 4
    if subclasses:
        push('Built-in subclasses:')
        for subclassname in subclasses[:MAX_SUBCLASSES_TO_DISPLAY]:
            push('    ' + subclassname)
        if no_of_subclasses > MAX_SUBCLASSES_TO_DISPLAY:
            push('    ... and ' + str(no_of_subclasses - MAX_SUBCLASSES_TO_DISPLAY) + ' other subclasses')
        push('')

    class HorizontalRule:

        def __init__(self):
            self.needone = 0

        def maybe(self):
            if self.needone:
                push('-' * 70)
            self.needone = 1
    hr = HorizontalRule()

    def spill(msg, attrs, predicate):
        ok, attrs = _split_list(attrs, predicate)
        if ok:
            hr.maybe()
            push(msg)
            for name, kind, homecls, value in ok:
                try:
                    value = getattr(object, name)
                except Exception:
                    push(self.docdata(value, name, mod))
                else:
                    push(self.document(value, name, mod, object))
        return attrs

    def spilldescriptors(msg, attrs, predicate):
        ok, attrs = _split_list(attrs, predicate)
        if ok:
            hr.maybe()
            push(msg)
            for name, kind, homecls, value in ok:
                push(self.docdata(value, name, mod))
        return attrs

    def spilldata(msg, attrs, predicate):
        ok, attrs = _split_list(attrs, predicate)
        if ok:
            hr.maybe()
            push(msg)
            for name, kind, homecls, value in ok:
                doc = getdoc(value)
                try:
                    obj = getattr(object, name)
                except AttributeError:
                    obj = homecls.__dict__[name]
                push(self.docother(obj, name, mod, maxlen=70, doc=doc) + '\n')
        return attrs
    attrs = [(name, kind, cls, value) for name, kind, cls, value in classify_class_attrs(object) if visiblename(name, obj=object)]
    while attrs:
        if mro:
            thisclass = mro.popleft()
        else:
            thisclass = attrs[0][2]
        attrs, inherited = _split_list(attrs, lambda t: t[2] is thisclass)
        if object is not builtins.object and thisclass is builtins.object:
            attrs = inherited
            continue
        elif thisclass is object:
            tag = 'defined here'
        else:
            tag = 'inherited from %s' % classname(thisclass, object.__module__)
        sort_attributes(attrs, object)
        attrs = spill('Methods %s:\n' % tag, attrs, lambda t: t[1] == 'method')
        attrs = spill('Class methods %s:\n' % tag, attrs, lambda t: t[1] == 'class method')
        attrs = spill('Static methods %s:\n' % tag, attrs, lambda t: t[1] == 'static method')
        attrs = spilldescriptors('Readonly properties %s:\n' % tag, attrs, lambda t: t[1] == 'readonly property')
        attrs = spilldescriptors('Data descriptors %s:\n' % tag, attrs, lambda t: t[1] == 'data descriptor')
        attrs = spilldata('Data and other attributes %s:\n' % tag, attrs, lambda t: t[1] == 'data')
        assert attrs == []
        attrs = inherited
    contents = '\n'.join(contents)
    if not contents:
        return title + '\n'
    return title + '\n' + self.indent(contents.rstrip(), ' |  ') + '\n'