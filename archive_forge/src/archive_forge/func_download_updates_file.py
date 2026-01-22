from __future__ import absolute_import, division, print_function
import os
import time
def download_updates_file(updates_expiration):
    updates_filename = 'jenkins-plugin-cache.json'
    updates_dir = os.path.expanduser('~/.ansible/tmp')
    updates_file = os.path.join(updates_dir, updates_filename)
    download_updates = True
    if not os.path.isdir(updates_dir):
        os.makedirs(updates_dir, 448)
    if os.path.isfile(updates_file):
        ts_file = os.stat(updates_file).st_mtime
        ts_now = time.time()
        if ts_now - ts_file < updates_expiration:
            download_updates = False
    return (updates_file, download_updates)