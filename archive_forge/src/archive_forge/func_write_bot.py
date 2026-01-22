from __future__ import annotations
import collections.abc as c
import datetime
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .config import (
from . import junit_xml
def write_bot(self, args: TestConfig) -> None:
    """Write results to a file for ansibullbot to consume."""
    docs = self.find_docs()
    message = self.format_title(help_link=docs)
    output = self.format_block()
    if self.messages:
        verified = all(((m.confidence or 0) >= 50 for m in self.messages))
    else:
        verified = False
    bot_data = dict(verified=verified, docs=docs, results=[dict(message=message, output=output)])
    if args.explain:
        return
    write_json_test_results(ResultType.BOT, self.create_result_name('.json'), bot_data)