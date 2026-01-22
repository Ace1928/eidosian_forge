import pytest
import gc
import weakref
import time
import os.path
@pytest.fixture()
def kivy_metrics():
    from kivy.context import Context
    from kivy.metrics import MetricsBase, Metrics
    from kivy._metrics import dispatch_pixel_scale
    context = Context(init=False)
    context['Metrics'] = MetricsBase()
    context.push()
    dispatch_pixel_scale()
    try:
        yield Metrics
    finally:
        context.pop()
        Metrics._set_cached_scaling()