from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task_util
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def get_disk_counters():
    """Retrieves disk I/O statistics for all disks.

  Adapted from the psutil module's psutil._pslinux.disk_io_counters:
    http://code.google.com/p/psutil/source/browse/trunk/psutil/_pslinux.py

  Originally distributed under under a BSD license.
  Original Copyright (c) 2009, Jay Loden, Dave Daeschler, Giampaolo Rodola.

  Returns:
    A dictionary containing disk names mapped to the disk counters from
    /disk/diskstats.
  """
    sector_size = 512
    partitions = _get_partitions()
    retdict = {}
    try:
        with files.FileReader('/proc/diskstats') as f:
            lines = f.readlines()
            for line in lines:
                values = line.split()[:11]
                _, _, name, reads, _, rbytes, rtime, writes, _, wbytes, wtime = values
                if name in partitions:
                    rbytes = int(rbytes) * sector_size
                    wbytes = int(wbytes) * sector_size
                    reads = int(reads)
                    writes = int(writes)
                    rtime = int(rtime)
                    wtime = int(wtime)
                    retdict[name] = (reads, writes, rbytes, wbytes, rtime, wtime)
    except files.Error:
        pass
    return retdict