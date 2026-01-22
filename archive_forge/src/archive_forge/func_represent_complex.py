from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_complex(self, data):
    if data.imag == 0.0:
        data = '%r' % data.real
    elif data.real == 0.0:
        data = '%rj' % data.imag
    elif data.imag > 0:
        data = '%r+%rj' % (data.real, data.imag)
    else:
        data = '%r%rj' % (data.real, data.imag)
    return self.represent_scalar('tag:yaml.org,2002:python/complex', data)