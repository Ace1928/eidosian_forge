from __future__ import annotations
import copy
import typing as t
from . import common
def end_foaf_agent(self) -> None:
    if self.flag_agent:
        self.flag_agent = False
    self._clean_found_objs()
    if self.foaf_name:
        self.foaf_name.pop()
    self.agent_feeds = []
    self.agent_lists = []
    self.agent_opps = []
    self.flag_agent = False
    self.flag_feed = False
    self.flag_opportunity = False