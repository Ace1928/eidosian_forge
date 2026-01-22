from __future__ import (absolute_import, division, print_function)
def report_message(path, line, column, code, message, messages):
    """Report message if not already reported.
        :type path: str
        :type line: int
        :type column: int
        :type code: str
        :type message: str
        :type messages: set[str]
        """
    message = '%s:%d:%d: %s: %s' % (path, line, column, code, message)
    if message not in messages:
        messages.add(message)
        print(message)