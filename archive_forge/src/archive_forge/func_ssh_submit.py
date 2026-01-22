from __future__ import absolute_import
from multiprocessing import Pool, Process
import os, subprocess, logging
from threading import Thread
from . import tracker
def ssh_submit(nworker, nserver, pass_envs):
    """
        customized submit script
        """

    def run(prog):
        subprocess.check_call(prog, shell=True)
    local_dir = os.getcwd() + '/'
    working_dir = local_dir
    if args.sync_dst_dir is not None and args.sync_dst_dir != 'None':
        working_dir = args.sync_dst_dir
        pool = Pool(processes=len(hosts))
        for h in hosts:
            pool.apply_async(sync_dir, args=(local_dir, h, working_dir))
        pool.close()
        pool.join()
    for i in range(nworker + nserver):
        pass_envs['DMLC_ROLE'] = 'server' if i < nserver else 'worker'
        node, port = hosts[i % len(hosts)]
        pass_envs['DMLC_NODE_HOST'] = node
        prog = get_env(pass_envs) + ' cd ' + working_dir + '; ' + ' '.join(args.command)
        prog = 'ssh -o StrictHostKeyChecking=no ' + node + ' -p ' + port + " '" + prog + "'"
        thread = Thread(target=run, args=(prog,))
        thread.setDaemon(True)
        thread.start()
    return ssh_submit