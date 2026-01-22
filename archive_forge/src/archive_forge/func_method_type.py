import sys
import types
def method_type(callable, instance, klass):
    return types.MethodType(callable, instance or klass())