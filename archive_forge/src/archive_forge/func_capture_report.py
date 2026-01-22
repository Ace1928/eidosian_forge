from __future__ import (absolute_import, division, print_function)
def capture_report(path, capture, messages):
    """Report on captured output.
        :type path: str
        :type capture: Capture
        :type messages: set[str]
        """
    capture.stdout.flush()
    capture.stderr.flush()
    stdout_value = capture.stdout.buffer.getvalue()
    if stdout_value:
        first = stdout_value.decode().strip().splitlines()[0].strip()
        report_message(path, 0, 0, 'stdout', first, messages)
    stderr_value = capture.stderr.buffer.getvalue()
    if stderr_value:
        first = stderr_value.decode().strip().splitlines()[0].strip()
        report_message(path, 0, 0, 'stderr', first, messages)