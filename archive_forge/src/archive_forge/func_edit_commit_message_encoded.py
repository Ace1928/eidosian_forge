import codecs
import os
import sys
from io import BytesIO, StringIO
from subprocess import call
from . import bedding, cmdline, config, osutils, trace, transport, ui
from .errors import BzrError
from .hooks import Hooks
def edit_commit_message_encoded(infotext, ignoreline=DEFAULT_IGNORE_LINE, start_message=None):
    """Let the user edit a commit message in a temp file.

    This is run if they don't give a message or
    message-containing file on the command line.

    :param infotext:    Text to be displayed at bottom of message
                        for the user's reference;
                        currently similar to 'bzr status'.
                        The string is already encoded

    :param ignoreline:  The separator to use above the infotext.

    :param start_message:   The text to place above the separator, if any.
                            This will not be removed from the message
                            after the user has edited it.
                            The string is already encoded

    :return:    commit message or None.
    """
    msgfilename = None
    try:
        msgfilename, hasinfo = _create_temp_file_with_commit_template(infotext, ignoreline, start_message)
        if not msgfilename:
            return None
        basename = osutils.basename(msgfilename)
        msg_transport = transport.get_transport_from_path(osutils.dirname(msgfilename))
        reference_content = msg_transport.get_bytes(basename)
        if not _run_editor(msgfilename):
            return None
        edited_content = msg_transport.get_bytes(basename)
        if edited_content == reference_content:
            if not ui.ui_factory.confirm_action('Commit message was not edited, use anyway', 'breezy.msgeditor.unchanged', {}):
                return ''
        started = False
        msg = []
        lastline, nlines = (0, 0)
        with codecs.open(msgfilename, mode='rb', encoding=osutils.get_user_encoding()) as f:
            try:
                for line in f:
                    stripped_line = line.strip()
                    if not started:
                        if stripped_line != '':
                            started = True
                        else:
                            continue
                    if hasinfo and stripped_line == ignoreline:
                        break
                    nlines += 1
                    if stripped_line != '':
                        lastline = nlines
                    msg.append(line)
            except UnicodeDecodeError:
                raise BadCommitMessageEncoding()
        if len(msg) == 0:
            return ''
        del msg[lastline:]
        if not msg[-1].endswith('\n'):
            return '{}{}'.format(''.join(msg), '\n')
        else:
            return ''.join(msg)
    finally:
        if msgfilename is not None:
            try:
                os.unlink(msgfilename)
            except OSError as e:
                trace.warning('failed to unlink %s: %s; ignored', msgfilename, e)