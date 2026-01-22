import os
import re
import time
import platform
import mimetypes
import subprocess
from os.path import join as pjoin
from collections import defaultdict
from libcloud.utils.py3 import ET, ensure_string
from libcloud.compute.base import Node, NodeState, NodeDriver
from libcloud.compute.types import Provider
from libcloud.utils.networking import is_public_subnet
def ex_take_node_screenshot(self, node, directory, screen=0):
    """
        Take a screenshot of a monitoring of a running instance.

        :param node: Node to take the screenshot of.
        :type node: :class:`libcloud.compute.base.Node`

        :param directory: Path where the screenshot will be saved.
        :type directory: ``str``

        :param screen: ID of the monitor to take the screenshot of.
        :type screen: ``int``

        :return: Full path where the screenshot has been saved.
        :rtype: ``str``
        """
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise ValueError('Invalid value for directory argument')
    domain = self._get_domain_for_node(node=node)
    stream = self.connection.newStream()
    mime_type = domain.screenshot(stream=stream, screen=0)
    extensions = mimetypes.guess_all_extensions(type=mime_type)
    if extensions:
        extension = extensions[0]
    else:
        extension = '.png'
    name = 'screenshot-{}{}'.format(int(time.time()), extension)
    file_path = pjoin(directory, name)
    with open(file_path, 'wb') as fp:

        def write(stream, buf, opaque):
            fp.write(buf)
        stream.recvAll(write, None)
    try:
        stream.finish()
    except Exception:
        pass
    return file_path