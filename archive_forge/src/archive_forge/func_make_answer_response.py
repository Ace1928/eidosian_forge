from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def make_answer_response(vm, answers):
    """Make the response contents to answer against locked a virtual machine.

    Args:
        vm: Virtual machine management object
        answers: Answer contents

    Returns: Dict with answer id and number
    Raises: TaskError on failure
    """
    response_list = {}
    for message in vm.runtime.question.message:
        response_list[message.id] = {}
        for choice in vm.runtime.question.choice.choiceInfo:
            response_list[message.id].update({choice.label: choice.key})
    responses = []
    try:
        for answer in answers:
            responses.append({'id': vm.runtime.question.id, 'response_num': response_list[answer['question']][answer['response']]})
    except Exception:
        raise TaskError('not found %s or %s or both in the response list' % (answer['question'], answer['response']))
    return responses