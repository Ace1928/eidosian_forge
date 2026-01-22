import collections
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.util import traceback_utils
def get_message(self):
    """Returns the message for the underlying exception."""
    lines = []
    lines.append('in user code:')
    lines.append('')
    for frame_info in reversed(self.translated_stack):
        if traceback_utils.is_traceback_filtering_enabled() and (not traceback_utils.include_frame(frame_info.filename)):
            continue
        formatted_line = f'    File "{frame_info.filename}", line {frame_info.lineno}, in {frame_info.function_name}'
        if frame_info.is_converted:
            formatted_line += '  *'
        elif frame_info.is_allowlisted:
            formatted_line += '  **'
        lines.append(formatted_line)
        if frame_info.code is None:
            code_snippet = '<source unavailable>'
        else:
            code_snippet = frame_info.code.strip()
        lines.append('        {}'.format(code_snippet))
    lines.append('')
    message_lines = self.cause_message.split('\n')
    for i in range(len(message_lines)):
        message_lines[i] = '    ' + message_lines[i]
    lines.extend(message_lines)
    lines.append('')
    return '\n'.join(lines)