import traceback
from io import StringIO
from java.lang import StringBuffer  # @UnresolvedImport
from java.lang import String  # @UnresolvedImport
import java.lang  # @UnresolvedImport
import sys
from _pydev_bundle._pydev_tipper_common import do_find
from org.python.core import PyReflectedFunction  # @UnresolvedImport
from org.python import core  # @UnresolvedImport
from org.python.core import PyClass  # @UnresolvedImport
import java.util
def dir_obj(obj):
    ret = []
    found = java.util.HashMap()
    original = obj
    if hasattr(obj, '__class__'):
        if obj.__class__ == java.lang.Class:
            classes = []
            classes.append(obj)
            try:
                c = obj.getSuperclass()
            except TypeError:
                c = obj.getSuperclass(obj)
            while c != None:
                classes.append(c)
                c = c.getSuperclass()
            interfs = []
            for obj in classes:
                try:
                    interfs.extend(obj.getInterfaces())
                except TypeError:
                    interfs.extend(obj.getInterfaces(obj))
            classes.extend(interfs)
            for obj in classes:
                try:
                    declaredMethods = obj.getDeclaredMethods()
                except TypeError:
                    declaredMethods = obj.getDeclaredMethods(obj)
                try:
                    declaredFields = obj.getDeclaredFields()
                except TypeError:
                    declaredFields = obj.getDeclaredFields(obj)
                for i in range(len(declaredMethods)):
                    name = declaredMethods[i].getName()
                    ret.append(name)
                    found.put(name, 1)
                for i in range(len(declaredFields)):
                    name = declaredFields[i].getName()
                    ret.append(name)
                    found.put(name, 1)
        elif isclass(obj.__class__):
            d = dir(obj.__class__)
            for name in d:
                ret.append(name)
                found.put(name, 1)
    d = dir(original)
    for name in d:
        if found.get(name) != 1:
            ret.append(name)
    return ret