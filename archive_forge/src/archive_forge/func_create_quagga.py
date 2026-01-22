import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def create_quagga(self, tagname='quagga', image=None, check_exist=False):
    if check_exist and self.exist(tagname):
        return tagname
    workdir = os.path.join(TEST_BASE_DIR, tagname)
    pkges = ' '.join(['telnet', 'tcpdump', 'quagga-bgpd'])
    if image:
        use_image = image
    else:
        use_image = self.baseimage
    c = CmdBuffer()
    c << 'FROM %s' % use_image
    c << 'RUN apt-get update'
    c << 'RUN apt-get install -qy --no-install-recommends %s' % pkges
    c << 'RUN echo "#!/bin/sh" > /bgpd'
    c << 'RUN echo mkdir -p /run/quagga >> /bgpd'
    c << 'RUN echo chmod 755 /run/quagga >> /bgpd'
    c << 'RUN echo chown quagga:quagga /run/quagga >> /bgpd'
    c << 'RUN echo exec /usr/sbin/bgpd >> /bgpd'
    c << 'RUN chmod +x /bgpd'
    c << 'CMD /bgpd'
    self.cmd.sudo('rm -rf %s' % workdir)
    self.cmd.execute('mkdir -p %s' % workdir)
    self.cmd.execute("echo '%s' > %s/Dockerfile" % (str(c), workdir))
    self.build(tagname, workdir)
    return tagname