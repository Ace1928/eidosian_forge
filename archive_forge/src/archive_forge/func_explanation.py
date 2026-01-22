import logging
import time
import jmespath
from botocore.docs.docstring import WaiterDocstring
from botocore.utils import get_service_module_name
from . import xform_name
from .exceptions import ClientError, WaiterConfigError, WaiterError
@property
def explanation(self):
    if self.matcher == 'path':
        return 'For expression "{}" we matched expected path: "{}"'.format(self.argument, self.expected)
    elif self.matcher == 'pathAll':
        return 'For expression "%s" all members matched excepted path: "%s"' % (self.argument, self.expected)
    elif self.matcher == 'pathAny':
        return 'For expression "%s" we matched expected path: "%s" at least once' % (self.argument, self.expected)
    elif self.matcher == 'status':
        return 'Matched expected HTTP status code: %s' % self.expected
    elif self.matcher == 'error':
        return 'Matched expected service error code: %s' % self.expected
    else:
        return 'No explanation for unknown waiter type: "%s"' % self.matcher