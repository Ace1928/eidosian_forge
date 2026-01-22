import codecs
import os
import sys
from io import BytesIO, StringIO
from subprocess import call
from . import bedding, cmdline, config, osutils, trace, transport, ui
from .errors import BzrError
from .hooks import Hooks
def generate_commit_message_template(commit, start_message=None):
    """Generate a commit message template.

    :param commit: Commit object for the active commit.
    :param start_message: Message to start with.
    :return: A start commit message or None for an empty start commit message.
    """
    start_message = None
    for hook in hooks['commit_message_template']:
        start_message = hook(commit, start_message)
    return start_message