import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from twisted.python import failure
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import UsageError
from scrapy.utils.conf import arglist_to_dict, feed_process_params_from_cli
def format_part_strings(self, part_strings):
    """
        Underline and title case command line help message headers.
        """
    if part_strings and part_strings[0].startswith('usage: '):
        part_strings[0] = 'Usage\n=====\n  ' + part_strings[0][len('usage: '):]
    headings = [i for i in range(len(part_strings)) if part_strings[i].endswith(':\n')]
    for index in headings[::-1]:
        char = '-' if 'Global Options' in part_strings[index] else '='
        part_strings[index] = part_strings[index][:-2].title()
        underline = ''.join(['\n', char * len(part_strings[index]), '\n'])
        part_strings.insert(index + 1, underline)
    return part_strings