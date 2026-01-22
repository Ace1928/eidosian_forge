import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from twisted.python import failure
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import UsageError
from scrapy.utils.conf import arglist_to_dict, feed_process_params_from_cli
def process_options(self, args, opts):
    ScrapyCommand.process_options(self, args, opts)
    try:
        opts.spargs = arglist_to_dict(opts.spargs)
    except ValueError:
        raise UsageError('Invalid -a value, use -a NAME=VALUE', print_help=False)
    if opts.output or opts.overwrite_output:
        feeds = feed_process_params_from_cli(self.settings, opts.output, opts.output_format, opts.overwrite_output)
        self.settings.set('FEEDS', feeds, priority='cmdline')