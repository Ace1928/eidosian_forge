from __future__ import (absolute_import, division, print_function)
import json
import os
import re
import email.utils
import smtplib
from ansible.module_utils.common.text.converters import to_bytes
from ansible.parsing.ajson import AnsibleJSONEncoder
from ansible.plugins.callback import CallbackBase
def body_blob(self, multiline, texttype):
    """ Turn some text output in a well-indented block for sending in a mail body """
    intro = 'with the following %s:\n\n' % texttype
    blob = ''
    for line in multiline.strip('\r\n').splitlines():
        blob += '%s\n' % line
    return intro + self.indent(blob) + '\n'