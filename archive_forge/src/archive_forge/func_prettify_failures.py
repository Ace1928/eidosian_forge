from kazoo import client
from kazoo import exceptions as k_exc
from oslo_utils import reflection
from oslo_utils import strutils
from taskflow import exceptions as exc
from taskflow import logging
def prettify_failures(failures, limit=-1):
    """Prettifies a checked commits failures (ignores sensitive data...)."""
    prettier = []
    for op, r in failures:
        pretty_op = reflection.get_class_name(op, fully_qualified=False)
        selected_attrs = ['path=%r' % op.path]
        try:
            if op.version != -1:
                selected_attrs.append('version=%s' % op.version)
        except AttributeError:
            pass
        pretty_op += '(%s)' % ', '.join(selected_attrs)
        pretty_cause = reflection.get_class_name(r, fully_qualified=False)
        prettier.append('%s@%s' % (pretty_cause, pretty_op))
    if limit <= 0 or len(prettier) <= limit:
        return ', '.join(prettier)
    else:
        leftover = prettier[limit:]
        prettier = prettier[0:limit]
        return ', '.join(prettier) + ' and %s more...' % len(leftover)