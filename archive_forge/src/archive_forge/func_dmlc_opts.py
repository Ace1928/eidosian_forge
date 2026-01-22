import argparse
import os, sys
import signal
import logging
def dmlc_opts(opts):
    """convert from mxnet's opts to dmlc's opts
    """
    args = ['--num-workers', str(opts.num_workers), '--num-servers', str(opts.num_servers), '--cluster', opts.launcher, '--host-file', opts.hostfile, '--sync-dst-dir', opts.sync_dst_dir]
    dopts = vars(opts)
    for key in ['env_server', 'env_worker', 'env']:
        for v in dopts[key]:
            args.append('--' + key.replace('_', '-'))
            args.append(v)
    args += opts.command
    try:
        from dmlc_tracker import opts
    except ImportError:
        print("Can't load dmlc_tracker package.  Perhaps you need to run")
        print('    git submodule update --init --recursive')
        raise
    dmlc_opts = opts.get_opts(args)
    return dmlc_opts