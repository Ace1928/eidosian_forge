import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def create_os_ken(self, tagname='os_ken', image=None, check_exist=False):
    if check_exist and self.exist(tagname):
        return tagname
    workdir = os.path.join(TEST_BASE_DIR, tagname)
    workdir_ctn = '/root/osrg/os_ken'
    pkges = ' '.join(['tcpdump', 'iproute2'])
    if image:
        use_image = image
    else:
        use_image = self.baseimage
    c = CmdBuffer()
    c << 'FROM %s' % use_image
    c << 'ADD os_ken %s' % workdir_ctn
    install = ' '.join(['RUN apt-get update', '&& apt-get install -qy --no-install-recommends %s' % pkges, '&& cd %s' % workdir_ctn, '&& rm -rf *.egg-info/ build/ dist/ .tox/ *.log&& pip install -r requirements.txt -r test-requirements.txt', '&& pip install .'])
    c << install
    self.cmd.sudo('rm -rf %s' % workdir)
    self.cmd.execute('mkdir -p %s' % workdir)
    self.cmd.execute("echo '%s' > %s/Dockerfile" % (str(c), workdir))
    self.cmd.execute('cp -r ../os_ken %s/' % workdir)
    self.build(tagname, workdir)
    return tagname