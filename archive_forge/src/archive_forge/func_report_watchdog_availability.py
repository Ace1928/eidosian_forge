from __future__ import annotations
from typing import Callable, Type, Union
import streamlit.watcher
from streamlit import cli_util, config, env_util
from streamlit.watcher.polling_path_watcher import PollingPathWatcher
def report_watchdog_availability():
    if not config.get_option('global.disableWatchdogWarning') and config.get_option('server.fileWatcherType') not in ['poll', 'none'] and (not _is_watchdog_available()):
        msg = '\n  $ xcode-select --install' if env_util.IS_DARWIN else ''
        cli_util.print_to_cli('  %s' % 'For better performance, install the Watchdog module:', fg='blue', bold=True)
        cli_util.print_to_cli('%s\n  $ pip install watchdog\n            ' % msg)