from __future__ import annotations
import json
import logging
import os
from typing import TYPE_CHECKING, Any
import tornado.websocket
from tornado import gen
from tornado.concurrent import run_on_executor
def log_terminal_output(self, log: str='') -> None:
    """
        Logs the terminal input/output
        :param log: log line to write
        :return:
        """
    self._logger.debug(log)