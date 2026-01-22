import sys
import subprocess
import shlex
import os
import argparse
import shutil
import logging
import coloredlogs
def build_components(components_source, concurrency):
    is_windows = sys.platform == 'win32'
    source_glob = components_source if components_source != 'all' else '{dash-core-components,dash-html-components,dash-table}'
    cmdstr = f"npx lerna exec --concurrency {concurrency} --scope='{source_glob}' -- npm run build"
    cmd = shlex.split(cmdstr, posix=not is_windows)
    status_print(cmdstr)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=is_windows) as proc:
        out, err = proc.communicate()
        status = proc.poll()
    if err:
        status_print(('🛑 ' if status else '') + err.decode(), file=sys.stderr)
    if status or not out:
        status_print(f'🚨 Finished updating component packages: {source_glob} (status={status}) 🚨', file=sys.stderr)
        sys.exit(1)
    if '{' in source_glob:
        source_glob = source_glob.split('{')[1].split('}')[0]
    for package in source_glob.split(','):
        build_directory = os.path.join('components', package, package.replace('-', '_').rstrip('/\\'))
        dest_dir = dest_dir_map.get(package) or package
        dest_path = os.path.join('dash', dest_dir)
        if not os.path.exists(dest_path):
            try:
                os.makedirs(dest_path)
            except OSError:
                logger.exception('🚨 Having issues manipulating %s', dest_path)
                sys.exit(1)
        if not os.path.exists(build_directory):
            status_print('🚨 Could not locate build artifacts.' + ' Check that the npm build process completed' + f' successfully for package: {package} 🚨')
            sys.exit(1)
        else:
            status_print(f'🚚 Moving build artifacts from {build_directory} to Dash 🚚')
            shutil.rmtree(dest_path)
            shutil.copytree(build_directory, dest_path)
            with open(os.path.join(dest_path, '.gitkeep'), 'w', encoding='utf-8'):
                pass
            status_print(f'🟢 Finished moving build artifacts from {build_directory} to Dash 🟢')