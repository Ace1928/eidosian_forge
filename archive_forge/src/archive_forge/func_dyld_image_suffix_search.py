import os
from ctypes.macholib.framework import framework_info
from ctypes.macholib.dylib import dylib_info
from itertools import *
def dyld_image_suffix_search(iterator, env=None):
    """For a potential path iterator, add DYLD_IMAGE_SUFFIX semantics"""
    suffix = dyld_image_suffix(env)
    if suffix is None:
        return iterator

    def _inject(iterator=iterator, suffix=suffix):
        for path in iterator:
            if path.endswith('.dylib'):
                yield (path[:-len('.dylib')] + suffix + '.dylib')
            else:
                yield (path + suffix)
            yield path
    return _inject()