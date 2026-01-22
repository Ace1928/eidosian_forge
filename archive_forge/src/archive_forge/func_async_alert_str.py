import functools
import re
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
def async_alert_str(*, terminal_columns: int, prompt: str, line: str, cursor_offset: int, alert_msg: str) -> str:
    """Calculate the desired string, including ANSI escape codes, for displaying an asynchronous alert message.

    :param terminal_columns: terminal width (number of columns)
    :param prompt: prompt that is displayed on the current line
    :param line: current contents of the Readline line buffer
    :param cursor_offset: the offset of the current cursor position within line
    :param alert_msg: the message to display to the user
    :return: the correct string so that the alert message appears to the user to be printed above the current line.
    """
    prompt_lines = prompt.splitlines() or ['']
    num_prompt_terminal_lines = 0
    for line in prompt_lines[:-1]:
        line_width = style_aware_wcswidth(line)
        num_prompt_terminal_lines += int(line_width / terminal_columns) + 1
    last_prompt_line = prompt_lines[-1]
    last_prompt_line_width = style_aware_wcswidth(last_prompt_line)
    input_width = last_prompt_line_width + style_aware_wcswidth(line)
    num_input_terminal_lines = int(input_width / terminal_columns) + 1
    cursor_input_offset = last_prompt_line_width + cursor_offset
    cursor_input_line = int(cursor_input_offset / terminal_columns) + 1
    terminal_str = ''
    if cursor_input_line != num_input_terminal_lines:
        terminal_str += Cursor.DOWN(num_input_terminal_lines - cursor_input_line)
    total_lines = num_prompt_terminal_lines + num_input_terminal_lines
    terminal_str += (clear_line() + Cursor.UP(1)) * (total_lines - 1)
    terminal_str += clear_line()
    terminal_str += '\r' + alert_msg
    return terminal_str