import sys
import subprocess
import shlex
import os
import argparse
import shutil
import logging
import coloredlogs
def bootstrap_components(components_source, concurrency, install_type):
    is_windows = sys.platform == 'win32'
    source_glob = components_source if components_source != 'all' else '{dash-core-components,dash-html-components,dash-table}'
    cmdstr = f"npx lerna exec --concurrency {concurrency} --scope='{source_glob}' -- npm {install_type}"
    cmd = shlex.split(cmdstr, posix=not is_windows)
    status_print(cmdstr)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=is_windows) as proc:
        out, err = proc.communicate()
        status = proc.poll()
    if err:
        status_print(('🛑 ' if status else '') + err.decode(), file=sys.stderr)
    if status or not out:
        status_print(f'🚨 Failed installing npm dependencies for component packages: {source_glob} (status={status}) 🚨', file=sys.stderr)
        sys.exit(1)
    else:
        status_print(f'🟢 Finished installing npm dependencies for component packages: {source_glob} 🟢', file=sys.stderr)