import codecs
import os
import pipes
import re
import subprocess
import tempfile
from humanfriendly.terminal import (
from coloredlogs.converter.colors import (
def select_text_color(r, g, b):
    """
    Choose a suitable color for the inverse text style.

    :param r: The amount of red (an integer between 0 and 255).
    :param g: The amount of green (an integer between 0 and 255).
    :param b: The amount of blue (an integer between 0 and 255).
    :returns: A CSS color in hexadecimal notation (a string).

    In inverse mode the color that is normally used for the text is instead
    used for the background, however this can render the text unreadable. The
    purpose of :func:`select_text_color()` is to make an effort to select a
    suitable text color. Based on http://stackoverflow.com/a/3943023/112731.
    """
    return '#000' if r * 0.299 + g * 0.587 + b * 0.114 > 186 else '#FFF'