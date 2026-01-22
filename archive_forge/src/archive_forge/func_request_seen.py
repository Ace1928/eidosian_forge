from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Set
from warnings import warn
from twisted.internet.defer import Deferred
from scrapy.http.request import Request
from scrapy.settings import BaseSettings
from scrapy.spiders import Spider
from scrapy.utils.deprecate import ScrapyDeprecationWarning
from scrapy.utils.job import job_dir
from scrapy.utils.request import (
def request_seen(self, request: Request) -> bool:
    fp = self.request_fingerprint(request)
    if fp in self.fingerprints:
        return True
    self.fingerprints.add(fp)
    if self.file:
        self.file.write(fp + '\n')
    return False