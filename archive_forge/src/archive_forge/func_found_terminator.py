import sys
import os
import errno
import getopt
import time
import socket
import collections
from warnings import _deprecated, warn
from email._header_value_parser import get_addr_spec, get_angle_addr
import asyncore
import asynchat
def found_terminator(self):
    line = self._emptystring.join(self.received_lines)
    print('Data:', repr(line), file=DEBUGSTREAM)
    self.received_lines = []
    if self.smtp_state == self.COMMAND:
        sz, self.num_bytes = (self.num_bytes, 0)
        if not line:
            self.push('500 Error: bad syntax')
            return
        if not self._decode_data:
            line = str(line, 'utf-8')
        i = line.find(' ')
        if i < 0:
            command = line.upper()
            arg = None
        else:
            command = line[:i].upper()
            arg = line[i + 1:].strip()
        max_sz = self.command_size_limits[command] if self.extended_smtp else self.command_size_limit
        if sz > max_sz:
            self.push('500 Error: line too long')
            return
        method = getattr(self, 'smtp_' + command, None)
        if not method:
            self.push('500 Error: command "%s" not recognized' % command)
            return
        method(arg)
        return
    else:
        if self.smtp_state != self.DATA:
            self.push('451 Internal confusion')
            self.num_bytes = 0
            return
        if self.data_size_limit and self.num_bytes > self.data_size_limit:
            self.push('552 Error: Too much mail data')
            self.num_bytes = 0
            return
        data = []
        for text in line.split(self._linesep):
            if text and text[0] == self._dotsep:
                data.append(text[1:])
            else:
                data.append(text)
        self.received_data = self._newline.join(data)
        args = (self.peer, self.mailfrom, self.rcpttos, self.received_data)
        kwargs = {}
        if not self._decode_data:
            kwargs = {'mail_options': self.mail_options, 'rcpt_options': self.rcpt_options}
        status = self.smtp_server.process_message(*args, **kwargs)
        self._set_post_data_state()
        if not status:
            self.push('250 OK')
        else:
            self.push(status)