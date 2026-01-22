from enum import Flag
def qt_class_flags(type):
    f = _QT_CLASS_FLAGS.get(type)
    return f if f else ClassFlag(0)