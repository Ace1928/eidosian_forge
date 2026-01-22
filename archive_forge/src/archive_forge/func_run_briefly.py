import asyncio
import asyncio.events
import collections
import contextlib
import gc
import logging
import os
import pprint
import re
import select
import socket
import ssl
import sys
import tempfile
import threading
import time
import unittest
import uvloop
def run_briefly(loop):

    async def once():
        pass
    gen = once()
    t = loop.create_task(gen)
    t._log_destroy_pending = False
    try:
        loop.run_until_complete(t)
    finally:
        gen.close()