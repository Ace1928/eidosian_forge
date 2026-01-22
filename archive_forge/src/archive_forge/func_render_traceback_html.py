from __future__ import annotations
import itertools
import linecache
import os
import re
import sys
import sysconfig
import traceback
import typing as t
from markupsafe import escape
from ..utils import cached_property
from .console import Console
def render_traceback_html(self, include_title: bool=True) -> str:
    library_frames = [f.is_library for f in self.all_frames]
    mark_library = 0 < sum(library_frames) < len(library_frames)
    rows = []
    if not library_frames:
        classes = 'traceback noframe-traceback'
    else:
        classes = 'traceback'
        for msg, current in reversed(self.all_tracebacks):
            row_parts = []
            if msg is not None:
                row_parts.append(f'<li><div class="exc-divider">{msg}:</div>')
            for frame in current.stack:
                frame = t.cast(DebugFrameSummary, frame)
                info = f' title="{escape(frame.info)}"' if frame.info else ''
                row_parts.append(f'<li{info}>{frame.render_html(mark_library)}')
            rows.append('\n'.join(row_parts))
    is_syntax_error = issubclass(self._te.exc_type, SyntaxError)
    if include_title:
        if is_syntax_error:
            title = 'Syntax Error'
        else:
            title = 'Traceback <em>(most recent call last)</em>:'
    else:
        title = ''
    exc_full = escape(''.join(self._te.format_exception_only()))
    if is_syntax_error:
        description = f'<pre class=syntaxerror>{exc_full}</pre>'
    else:
        description = f'<blockquote>{exc_full}</blockquote>'
    return SUMMARY_HTML % {'classes': classes, 'title': f'<h3>{title}</h3>', 'frames': '\n'.join(rows), 'description': description}