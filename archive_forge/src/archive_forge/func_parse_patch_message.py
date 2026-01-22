import email.parser
import time
from difflib import SequenceMatcher
from typing import BinaryIO, Optional, TextIO, Union
from .objects import S_ISGITLINK, Blob, Commit
from .pack import ObjectContainer
def parse_patch_message(msg, encoding=None):
    """Extract a Commit object and patch from an e-mail message.

    Args:
      msg: An email message (email.message.Message)
      encoding: Encoding to use to encode Git commits
    Returns: Tuple with commit object, diff contents and git version
    """
    c = Commit()
    c.author = msg['from'].encode(encoding)
    c.committer = msg['from'].encode(encoding)
    try:
        patch_tag_start = msg['subject'].index('[PATCH')
    except ValueError:
        subject = msg['subject']
    else:
        close = msg['subject'].index('] ', patch_tag_start)
        subject = msg['subject'][close + 2:]
    c.message = (subject.replace('\n', '') + '\n').encode(encoding)
    first = True
    body = msg.get_payload(decode=True)
    lines = body.splitlines(True)
    line_iter = iter(lines)
    for line in line_iter:
        if line == b'---\n':
            break
        if first:
            if line.startswith(b'From: '):
                c.author = line[len(b'From: '):].rstrip()
            else:
                c.message += b'\n' + line
            first = False
        else:
            c.message += line
    diff = b''
    for line in line_iter:
        if line == b'-- \n':
            break
        diff += line
    try:
        version = next(line_iter).rstrip(b'\n')
    except StopIteration:
        version = None
    return (c, diff, version)