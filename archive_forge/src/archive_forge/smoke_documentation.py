import asyncio
import sqlite3
from pathlib import Path
from sqlite3 import OperationalError
from threading import Thread
from unittest import IsolatedAsyncioTestCase as TestCase, SkipTest
import aiosqlite
from .helpers import setup_logger
Assert that after creating a deterministic custom function, it can be used.

        https://sqlite.org/deterministic.html
        