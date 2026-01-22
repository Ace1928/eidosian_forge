from __future__ import print_function
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty, IntegerProperty, ListProperty, ReferenceProperty, CalculatedProperty
from boto.manage.server import Server
from boto.manage import propget
import boto.utils
import boto.ec2
import time
import traceback
from contextlib import closing
import datetime
def get_snapshots(self):
    """
        Returns a list of all completed snapshots for this volume ID.
        """
    ec2 = self.get_ec2_connection()
    rs = ec2.get_all_snapshots()
    all_vols = [self.volume_id] + self.past_volume_ids
    snaps = []
    for snapshot in rs:
        if snapshot.volume_id in all_vols:
            if snapshot.progress == '100%':
                snapshot.date = boto.utils.parse_ts(snapshot.start_time)
                snapshot.keep = True
                snaps.append(snapshot)
    snaps.sort(cmp=lambda x, y: cmp(x.date, y.date))
    return snaps