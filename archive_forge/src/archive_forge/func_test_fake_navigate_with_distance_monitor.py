from minerl.herobraine.hero import handlers
from typing import List
from minerl.herobraine.hero.handlers.translation import TranslationHandler
import time
from minerl.herobraine.env_specs.navigate_specs import Navigate
import coloredlogs
import logging
def test_fake_navigate_with_distance_monitor():
    task = NavigateWithDistanceMonitor(dense=True, extreme=False)
    fake_env = task.make(fake=True)
    _ = fake_env.reset()
    for _ in range(100):
        fake_obs, _, _, fake_monitor = fake_env.step(fake_env.action_space.sample())
        assert fake_monitor in fake_env.monitor_space
        assert 'compass' in fake_monitor
        assert 'distance' in fake_monitor['compass']