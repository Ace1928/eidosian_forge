from datetime import datetime, timedelta
from itertools import islice
from operator import itemgetter
from typing import Any, Dict, List, cast
from docutils import nodes
import sphinx
from sphinx.application import Sphinx
from sphinx.domains import Domain
from sphinx.locale import __
from sphinx.util import logging
def on_doctree_read(app: Sphinx, doctree: nodes.document) -> None:
    """Record a reading duration."""
    started_at = app.env.temp_data.get('started_at')
    duration = datetime.now() - started_at
    domain = cast(DurationDomain, app.env.get_domain('duration'))
    domain.note_reading_duration(duration)