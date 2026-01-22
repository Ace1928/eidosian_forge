from minerl.herobraine.hero import handlers
from typing import List
from minerl.herobraine.hero.handlers.translation import TranslationHandler
import time
from minerl.herobraine.env_specs.navigate_specs import Navigate
import coloredlogs
import logging
def test_fake_navigate():
    _test_fake_env(Navigate(dense=True, extreme=False))
    _test_fake_env(Navigate(dense=True, extreme=False, agent_count=3))