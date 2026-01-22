import errno
import io
import logging
import logging.handlers
import os
import queue
import re
import struct
import threading
import traceback
from socketserver import ThreadingTCPServer, StreamRequestHandler
def configure_handler(self, config):
    """Configure a handler from a dictionary."""
    config_copy = dict(config)
    formatter = config.pop('formatter', None)
    if formatter:
        try:
            formatter = self.config['formatters'][formatter]
        except Exception as e:
            raise ValueError('Unable to set formatter %r' % formatter) from e
    level = config.pop('level', None)
    filters = config.pop('filters', None)
    if '()' in config:
        c = config.pop('()')
        if not callable(c):
            c = self.resolve(c)
        factory = c
    else:
        cname = config.pop('class')
        klass = self.resolve(cname)
        if issubclass(klass, logging.handlers.MemoryHandler) and 'target' in config:
            try:
                th = self.config['handlers'][config['target']]
                if not isinstance(th, logging.Handler):
                    config.update(config_copy)
                    raise TypeError('target not configured yet')
                config['target'] = th
            except Exception as e:
                raise ValueError('Unable to set target handler %r' % config['target']) from e
        elif issubclass(klass, logging.handlers.SMTPHandler) and 'mailhost' in config:
            config['mailhost'] = self.as_tuple(config['mailhost'])
        elif issubclass(klass, logging.handlers.SysLogHandler) and 'address' in config:
            config['address'] = self.as_tuple(config['address'])
        factory = klass
    kwargs = {k: config[k] for k in config if k != '.' and valid_ident(k)}
    try:
        result = factory(**kwargs)
    except TypeError as te:
        if "'stream'" not in str(te):
            raise
        kwargs['strm'] = kwargs.pop('stream')
        result = factory(**kwargs)
    if formatter:
        result.setFormatter(formatter)
    if level is not None:
        result.setLevel(logging._checkLevel(level))
    if filters:
        self.add_filters(result, filters)
    props = config.pop('.', None)
    if props:
        for name, value in props.items():
            setattr(result, name, value)
    return result