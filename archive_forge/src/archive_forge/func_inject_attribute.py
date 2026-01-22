import sys
from collections import namedtuple
def inject_attribute(class_attributes, name, value):
    if name in class_attributes:
        raise RuntimeError('Cannot inject class attribute "%s", attribute already exists in class dict.' % name)
    else:
        class_attributes[name] = value