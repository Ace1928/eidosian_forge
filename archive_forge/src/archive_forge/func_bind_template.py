from contextlib import contextmanager
from copy import copy
@contextmanager
def bind_template(self, template):
    if self.template is not None:
        raise RuntimeError('Context is already bound to a template')
    self.template = template
    processors = template.engine.template_context_processors + self._processors
    updates = {}
    for processor in processors:
        context = processor(self.request)
        try:
            updates.update(context)
        except TypeError as e:
            raise TypeError(f"Context processor {processor.__qualname__} didn't return a dictionary.") from e
    self.dicts[self._processors_index] = updates
    try:
        yield
    finally:
        self.template = None
        self.dicts[self._processors_index] = {}