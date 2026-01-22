import io
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from cProfile import Profile
from functools import wraps
from ..config import config
from ..util import escape
from .state import state
def get_profiles(profilers, **kwargs):
    from ..pane import HTML, Markdown
    profiles = []
    for (path, engine), sessions in profilers.items():
        if not sessions:
            continue
        if engine == 'memray':
            src, style = render_memray(path, sessions, **kwargs)
            if kwargs.get('reporter', 'tree') not in ('flamegraph', 'table'):
                from ..widgets import Terminal
                term = Terminal(sizing_mode='stretch_both', margin=0, min_height=600)
                term.write(src)
                profiles.append((path, term))
                continue
            else:
                src = escape(src)
        if engine == 'pyinstrument':
            src, style = render_pyinstrument(sessions, **kwargs)
        elif engine == 'snakeviz':
            src, style = render_snakeviz(path, sessions)
        html = HTML(f'<iframe srcdoc="{src}" width="100%" height="100%" frameBorder="0" style="{style}"></iframe>', sizing_mode='stretch_both', margin=0, min_height=800)
        profiles.append((path, html))
    if not profiles:
        profiles.append(('', Markdown('No profiling output available')))
    return profiles