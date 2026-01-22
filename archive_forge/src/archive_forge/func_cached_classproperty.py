import sys
def cached_classproperty(func):
    return _CachedClassProperty(func)