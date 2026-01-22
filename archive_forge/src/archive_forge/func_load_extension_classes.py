import os
import sys
import warnings
from contextlib import contextmanager
from importlib import import_module, reload
from kombu.utils.imports import symbol_by_name
def load_extension_classes(namespace):
    for name, class_name in load_extension_class_names(namespace):
        try:
            cls = symbol_by_name(class_name)
        except (ImportError, SyntaxError) as exc:
            warnings.warn(f'Cannot load {namespace} extension {class_name!r}: {exc!r}')
        else:
            yield (name, cls)