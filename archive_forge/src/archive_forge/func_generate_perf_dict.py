from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_perf_dict(blade):
    perf_info = {}
    total_perf = blade.arrays.list_arrays_performance()
    http_perf = blade.arrays.list_arrays_performance(protocol='http')
    s3_perf = blade.arrays.list_arrays_performance(protocol='s3')
    nfs_perf = blade.arrays.list_arrays_performance(protocol='nfs')
    perf_info['aggregate'] = {'bytes_per_op': total_perf.items[0].bytes_per_op, 'bytes_per_read': total_perf.items[0].bytes_per_read, 'bytes_per_write': total_perf.items[0].bytes_per_write, 'read_bytes_per_sec': total_perf.items[0].read_bytes_per_sec, 'reads_per_sec': total_perf.items[0].reads_per_sec, 'usec_per_other_op': total_perf.items[0].usec_per_other_op, 'usec_per_read_op': total_perf.items[0].usec_per_read_op, 'usec_per_write_op': total_perf.items[0].usec_per_write_op, 'write_bytes_per_sec': total_perf.items[0].write_bytes_per_sec, 'writes_per_sec': total_perf.items[0].writes_per_sec}
    perf_info['http'] = {'bytes_per_op': http_perf.items[0].bytes_per_op, 'bytes_per_read': http_perf.items[0].bytes_per_read, 'bytes_per_write': http_perf.items[0].bytes_per_write, 'read_bytes_per_sec': http_perf.items[0].read_bytes_per_sec, 'reads_per_sec': http_perf.items[0].reads_per_sec, 'usec_per_other_op': http_perf.items[0].usec_per_other_op, 'usec_per_read_op': http_perf.items[0].usec_per_read_op, 'usec_per_write_op': http_perf.items[0].usec_per_write_op, 'write_bytes_per_sec': http_perf.items[0].write_bytes_per_sec, 'writes_per_sec': http_perf.items[0].writes_per_sec}
    perf_info['s3'] = {'bytes_per_op': s3_perf.items[0].bytes_per_op, 'bytes_per_read': s3_perf.items[0].bytes_per_read, 'bytes_per_write': s3_perf.items[0].bytes_per_write, 'read_bytes_per_sec': s3_perf.items[0].read_bytes_per_sec, 'reads_per_sec': s3_perf.items[0].reads_per_sec, 'usec_per_other_op': s3_perf.items[0].usec_per_other_op, 'usec_per_read_op': s3_perf.items[0].usec_per_read_op, 'usec_per_write_op': s3_perf.items[0].usec_per_write_op, 'write_bytes_per_sec': s3_perf.items[0].write_bytes_per_sec, 'writes_per_sec': s3_perf.items[0].writes_per_sec}
    perf_info['nfs'] = {'bytes_per_op': nfs_perf.items[0].bytes_per_op, 'bytes_per_read': nfs_perf.items[0].bytes_per_read, 'bytes_per_write': nfs_perf.items[0].bytes_per_write, 'read_bytes_per_sec': nfs_perf.items[0].read_bytes_per_sec, 'reads_per_sec': nfs_perf.items[0].reads_per_sec, 'usec_per_other_op': nfs_perf.items[0].usec_per_other_op, 'usec_per_read_op': nfs_perf.items[0].usec_per_read_op, 'usec_per_write_op': nfs_perf.items[0].usec_per_write_op, 'write_bytes_per_sec': nfs_perf.items[0].write_bytes_per_sec, 'writes_per_sec': nfs_perf.items[0].writes_per_sec}
    api_version = blade.api_version.list_versions().versions
    if REPLICATION_API_VERSION in api_version:
        file_repl_perf = blade.array_connections.list_array_connections_performance_replication(type='file-system')
        obj_repl_perf = blade.array_connections.list_array_connections_performance_replication(type='object-store')
        if len(file_repl_perf.total):
            perf_info['file_replication'] = {'received_bytes_per_sec': file_repl_perf.total[0].periodic.received_bytes_per_sec, 'transmitted_bytes_per_sec': file_repl_perf.total[0].periodic.transmitted_bytes_per_sec}
        if len(obj_repl_perf.total):
            perf_info['object_replication'] = {'received_bytes_per_sec': obj_repl_perf.total[0].periodic.received_bytes_per_sec, 'transmitted_bytes_per_sec': obj_repl_perf.total[0].periodic.transmitted_bytes_per_sec}
    return perf_info