import os
import re
import subprocess
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def expand_hostlist(hostlist):
    """Create a list of hosts out of a SLURM hostlist.

  The order of nodes is preserved and no deduplication is done
  Input: 'n[1-2],m5,o[3-4,6,7-9]')
  Output: ['n1', 'n2', 'm5', 'o3', 'o4', 'o6', 'o7', 'o8', 'o9']
  """

    def split_hostlist(hostlist):
        """Split hostlist at commas outside of range expressions ('[3-5]')."""
        in_brackets = False
        cur_host = ''
        for c in hostlist:
            if in_brackets:
                assert c != '['
                if c == ']':
                    in_brackets = False
            elif c == '[':
                in_brackets = True
            elif c == ',':
                assert cur_host != ''
                yield cur_host
                cur_host = ''
                continue
            cur_host += c
        if cur_host:
            yield cur_host

    def expand_range_expression(range_exp):
        """Expand a range expression like '3-5' to values 3,4,5."""
        for part in range_exp.split(','):
            sub_range = part.split('-')
            if len(sub_range) == 1:
                sub_range = sub_range * 2
            else:
                assert len(sub_range) == 2
            num_digits = len(sub_range[0])
            for i in range(int(sub_range[0]), int(sub_range[1]) + 1):
                yield str(i).zfill(num_digits)
    hosts = []
    try:
        for part in split_hostlist(hostlist):
            m = re.match('([^,[\\]]*)(\\[([^\\]]+)\\])?$', part)
            if m is None:
                raise ValueError('Invalid part: %s' % part)
            prefix = m.group(1) or ''
            if m.group(3) is None:
                hosts.append(prefix)
            else:
                hosts.extend((prefix + i for i in expand_range_expression(m.group(3))))
    except Exception as e:
        raise ValueError('Invalid hostlist format "%s": %s' % (hostlist, e))
    return hosts