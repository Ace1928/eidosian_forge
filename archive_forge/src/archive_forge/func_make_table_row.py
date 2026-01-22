import textwrap
import numpy as np
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.text import metadata
def make_table_row(contents, tag='td'):
    """Given an iterable of string contents, make a table row.

    Args:
      contents: An iterable yielding strings.
      tag: The tag to place contents in. Defaults to 'td', you might want 'th'.

    Returns:
      A string containing the content strings, organized into a table row.

    Example: make_table_row(['one', 'two', 'three']) == '''
    <tr>
    <td>one</td>
    <td>two</td>
    <td>three</td>
    </tr>'''
    """
    columns = ('<%s>%s</%s>\n' % (tag, s, tag) for s in contents)
    return '<tr>\n' + ''.join(columns) + '</tr>\n'