import functools
from oslo_utils import reflection
@functools.wraps(f)
def ins_wrap(self):
    if not set(dialects).issubset(ALLOWED_DIALECTS):
        raise ValueError('Please use allowed dialects: %s' % ALLOWED_DIALECTS)
    if self.engine.name not in dialects:
        msg = 'The test "%s" can be run only on %s. Current engine is %s.'
        args = (reflection.get_callable_name(f), ', '.join(dialects), self.engine.name)
        self.skipTest(msg % args)
    else:
        return f(self)