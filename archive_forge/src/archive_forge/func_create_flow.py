import contextlib
import hashlib
import logging
import os
import random
import sys
import time
import futurist
from oslo_utils import uuidutils
from taskflow import engines
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import models
from taskflow import task
import example_utils as eu  # noqa
def create_flow():
    flow = lf.Flow('root').add(PrintText('Starting vm creation.', no_slow=True), lf.Flow('vm-maker').add(DefineVMSpec('define_spec'), gf.Flow('img-maker').add(LocateImages('locate_images'), DownloadImages('download_images')), gf.Flow('net-maker').add(AllocateIP('get_my_ips'), CreateNetworkTpl('fetch_net_settings'), WriteNetworkSettings('write_net_settings')), gf.Flow('volume-maker').add(AllocateVolumes('allocate_my_volumes', provides='volumes'), FormatVolumes('volume_formatter')), BootVM('boot-it')), PrintText('Finished vm create.', no_slow=True), PrintText('Instance is running!', no_slow=True))
    return flow