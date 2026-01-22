import unittest
import oslo_i18n
from oslo_i18n import _lazy
def test_toggle_lazy(self):
    original = _lazy.USE_LAZY
    try:
        oslo_i18n.enable_lazy(True)
        oslo_i18n.enable_lazy(False)
    finally:
        oslo_i18n.enable_lazy(original)