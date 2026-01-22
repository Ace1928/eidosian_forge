from celery import bootsteps
from celery.utils.log import get_logger
from .events import Events
def on_clock_event(self, c, clock):
    c.app.clock.adjust(clock) if clock else c.app.clock.forward()