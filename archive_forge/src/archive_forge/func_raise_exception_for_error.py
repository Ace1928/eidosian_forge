from typing import Dict, Type
from libcloud.utils.py3 import httplib, xmlrpclib
from libcloud.common.base import Response, Connection
from libcloud.common.types import LibcloudError
def raise_exception_for_error(self, error_code, message):
    exceptionCls = self.exceptions.get(error_code, None)
    if exceptionCls is None:
        return
    context = self.connection.context
    driver = self.connection.driver
    params = {}
    if hasattr(exceptionCls, 'kwargs'):
        for key in exceptionCls.kwargs:
            if key in context:
                params[key] = context[key]
    raise exceptionCls(value=message, driver=driver, **params)