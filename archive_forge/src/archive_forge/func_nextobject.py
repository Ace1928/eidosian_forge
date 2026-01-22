import logging
import re
from typing import (
from . import settings
from .utils import choplist
def nextobject(self) -> PSStackEntry[ExtraT]:
    """Yields a list of objects.

        Arrays and dictionaries are represented as Python lists and
        dictionaries.

        :return: keywords, literals, strings, numbers, arrays and dictionaries.
        """
    while not self.results:
        pos, token = self.nexttoken()
        if isinstance(token, (int, float, bool, str, bytes, PSLiteral)):
            self.push((pos, token))
        elif token == KEYWORD_ARRAY_BEGIN:
            self.start_type(pos, 'a')
        elif token == KEYWORD_ARRAY_END:
            try:
                self.push(self.end_type('a'))
            except PSTypeError:
                if settings.STRICT:
                    raise
        elif token == KEYWORD_DICT_BEGIN:
            self.start_type(pos, 'd')
        elif token == KEYWORD_DICT_END:
            try:
                pos, objs = self.end_type('d')
                if len(objs) % 2 != 0:
                    error_msg = 'Invalid dictionary construct: %r' % objs
                    raise PSSyntaxError(error_msg)
                d = {literal_name(k): v for k, v in choplist(2, objs) if v is not None}
                self.push((pos, d))
            except PSTypeError:
                if settings.STRICT:
                    raise
        elif token == KEYWORD_PROC_BEGIN:
            self.start_type(pos, 'p')
        elif token == KEYWORD_PROC_END:
            try:
                self.push(self.end_type('p'))
            except PSTypeError:
                if settings.STRICT:
                    raise
        elif isinstance(token, PSKeyword):
            log.debug('do_keyword: pos=%r, token=%r, stack=%r', pos, token, self.curstack)
            self.do_keyword(pos, token)
        else:
            log.error('unknown token: pos=%r, token=%r, stack=%r', pos, token, self.curstack)
            self.do_keyword(pos, token)
            raise
        if self.context:
            continue
        else:
            self.flush()
    obj = self.results.pop(0)
    try:
        log.debug('nextobject: %r', obj)
    except Exception:
        log.debug('nextobject: (unprintable object)')
    return obj