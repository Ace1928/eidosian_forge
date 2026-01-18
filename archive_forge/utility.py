import json
import os
import logging
from typing import Dict, Any
from eidos_config import EidosConfig

UNIVERSAL_CONFIG_DIR = os.path.join(EidosConfig().base_dir, "environment")
UNIVERSAL_CONFIG_PATH = os.path.join(UNIVERSAL_CONFIG_DIR, "universal_config.json")


def ensure_config_dir_exists() -> None:
    """Ensures the universal configuration directory exists."""
    os.makedirs(UNIVERSAL_CONFIG_DIR, exist_ok=True)


def load_config_from_disk() -> Dict[str, Any]:
    """Loads universal configuration from disk or returns defaults. 💾"""
    ensure_config_dir_exists()
    if os.path.exists(UNIVERSAL_CONFIG_PATH):
        try:
            with open(UNIVERSAL_CONFIG_PATH, "r", encoding="utf-8") as config_file:
                return json.load(config_file)
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from: {UNIVERSAL_CONFIG_PATH}")
            return {"profiler": {}, "monitor": {}, "formatter": {}}
    return {"profiler": {}, "monitor": {}, "formatter": {}}


def save_config_to_disk(config: Dict[str, Any]) -> None:
    """Saves the universal configuration to disk. 💾"""
    ensure_config_dir_exists()
    with open(UNIVERSAL_CONFIG_PATH, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=4)
