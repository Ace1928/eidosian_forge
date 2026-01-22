import logging
from tornado import web
from . import BaseApiHandler
def error_reason(self, workername, response):
    """extracts error message from response"""
    for res in response:
        try:
            return res[workername].get('error', 'Unknown reason')
        except KeyError:
            pass
    logger.error("Failed to extract error reason from '%s'", response)
    return 'Unknown reason'