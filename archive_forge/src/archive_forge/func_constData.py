import is finished. But that is no problem since the module is passed in.
import warnings
from textwrap import dedent
def constData(self):
    cls = self.__class__
    name = cls.__qualname__
    warnings.warn(dedent(f'\n        {name}.constData is unpythonic and will be removed in Qt For Python 6.0 .\n        Please use {name}.data instead.'), PySideDeprecationWarningRemovedInQt6, stacklevel=2)
    return cls.data(self)