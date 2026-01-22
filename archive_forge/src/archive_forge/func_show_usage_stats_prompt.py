import json
import logging
import threading
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.usage.usage_constants as usage_constant
from ray.experimental.internal_kv import _internal_kv_initialized, _internal_kv_put
from ray.core.generated import usage_pb2, gcs_pb2
def show_usage_stats_prompt(cli: bool) -> None:
    if not usage_stats_prompt_enabled():
        return
    from ray.autoscaler._private.cli_logger import cli_logger
    prompt_print = cli_logger.print if cli else print
    usage_stats_enabledness = _usage_stats_enabledness()
    if usage_stats_enabledness is UsageStatsEnabledness.DISABLED_EXPLICITLY:
        prompt_print(usage_constant.USAGE_STATS_DISABLED_MESSAGE)
    elif usage_stats_enabledness is UsageStatsEnabledness.ENABLED_BY_DEFAULT:
        if not cli:
            prompt_print(usage_constant.USAGE_STATS_ENABLED_BY_DEFAULT_FOR_RAY_INIT_MESSAGE)
        elif cli_logger.interactive:
            enabled = cli_logger.confirm(False, usage_constant.USAGE_STATS_CONFIRMATION_MESSAGE, _default=True, _timeout_s=10)
            set_usage_stats_enabled_via_env_var(enabled)
            try:
                set_usage_stats_enabled_via_config(enabled)
            except Exception as e:
                logger.debug(f'Failed to persist usage stats choice for future clusters: {e}')
            if enabled:
                prompt_print(usage_constant.USAGE_STATS_ENABLED_FOR_CLI_MESSAGE)
            else:
                prompt_print(usage_constant.USAGE_STATS_DISABLED_MESSAGE)
        else:
            prompt_print(usage_constant.USAGE_STATS_ENABLED_BY_DEFAULT_FOR_CLI_MESSAGE)
    else:
        assert usage_stats_enabledness is UsageStatsEnabledness.ENABLED_EXPLICITLY
        prompt_print(usage_constant.USAGE_STATS_ENABLED_FOR_CLI_MESSAGE if cli else usage_constant.USAGE_STATS_ENABLED_FOR_RAY_INIT_MESSAGE)