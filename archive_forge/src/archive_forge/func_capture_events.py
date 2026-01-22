import curses
import sys
import threading
from datetime import datetime
from itertools import count
from math import ceil
from textwrap import wrap
from time import time
from celery import VERSION_BANNER, states
from celery.app import app_or_default
from celery.utils.text import abbr, abbrtask
def capture_events(app, state, display):

    def on_connection_error(exc, interval):
        print('Connection Error: {!r}.  Retry in {}s.'.format(exc, interval), file=sys.stderr)
    while 1:
        print('-> evtop: starting capture...', file=sys.stderr)
        with app.connection_for_read() as conn:
            try:
                conn.ensure_connection(on_connection_error, app.conf.broker_connection_max_retries)
                recv = app.events.Receiver(conn, handlers={'*': state.event})
                display.resetscreen()
                display.init_screen()
                recv.capture()
            except conn.connection_errors + conn.channel_errors as exc:
                print(f'Connection lost: {exc!r}', file=sys.stderr)