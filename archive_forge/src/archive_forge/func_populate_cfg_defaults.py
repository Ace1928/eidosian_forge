import enum
import logging
import logging.handlers
import asyncio
import socket
import queue
import argparse
import yaml
from . import defaults
def populate_cfg_defaults(cfg):
    if not cfg:
        cfg = {}
    if cfg.get('path') is None:
        cfg['host'] = cfg.get('host', defaults.HOST)
        cfg['port'] = cfg.get('port', defaults.PORT)
    cfg['reuse_port'] = cfg.get('reuse_port', defaults.REUSE_PORT)
    cfg['shutdown_timeout'] = cfg.get('shutdown_timeout', defaults.SHUTDOWN_TIMEOUT)
    cfg['cache_grace'] = cfg.get('cache_grace', defaults.CACHE_GRACE)
    if 'proactive_policy_fetching' not in cfg:
        cfg['proactive_policy_fetching'] = {}
    cfg['proactive_policy_fetching']['enabled'] = cfg['proactive_policy_fetching'].get('enabled', defaults.PROACTIVE_FETCH_ENABLED)
    cfg['proactive_policy_fetching']['interval'] = cfg['proactive_policy_fetching'].get('interval', defaults.PROACTIVE_FETCH_INTERVAL)
    cfg['proactive_policy_fetching']['concurrency_limit'] = cfg['proactive_policy_fetching'].get('concurrency_limit', defaults.PROACTIVE_FETCH_CONCURRENCY_LIMIT)
    cfg['proactive_policy_fetching']['grace_ratio'] = cfg['proactive_policy_fetching'].get('grace_ratio', defaults.PROACTIVE_FETCH_GRACE_RATIO)
    if 'cache' not in cfg:
        cfg['cache'] = {}
    cfg['cache']['type'] = cfg['cache'].get('type', defaults.CACHE_BACKEND)
    if cfg['cache']['type'] == 'internal':
        if 'options' not in cfg['cache']:
            cfg['cache']['options'] = {}
        cfg['cache']['options']['cache_size'] = cfg['cache']['options'].get('cache_size', defaults.INTERNAL_CACHE_SIZE)

    def populate_zone(zone):
        zone['timeout'] = zone.get('timeout', defaults.TIMEOUT)
        zone['strict_testing'] = zone.get('strict_testing', defaults.STRICT_TESTING)
        zone['require_sni'] = zone.get('require_sni', defaults.REQUIRE_SNI)
        return zone
    if 'default_zone' not in cfg:
        cfg['default_zone'] = {}
    populate_zone(cfg['default_zone'])
    if 'zones' not in cfg:
        cfg['zones'] = {}
    for zone in cfg['zones'].values():
        populate_zone(zone)
    return cfg