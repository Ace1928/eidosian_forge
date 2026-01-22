import asyncio
import copy
import io
import os
import sys
import IPython
import traitlets
import ipyvuetify as v
def has_event_loop():
    try:
        asyncio.get_event_loop()
        return True
    except RuntimeError:
        return False