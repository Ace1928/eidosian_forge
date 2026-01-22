import enum
import logging
import os
import types
import typing
@property
def extern_python_def(self) -> str:
    return 'extern "Python" {} {}({});'.format(self.rtype, self.name, ' ,'.join(self.arguments))