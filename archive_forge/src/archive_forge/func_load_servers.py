import os
import re
import logging
import collections
import pyzor.account
def load_servers(filepath):
    """Load the servers file."""
    logger = logging.getLogger('pyzor')
    if not os.path.exists(filepath):
        servers = []
    else:
        servers = []
        with open(filepath) as serverf:
            for line in serverf:
                line = line.strip()
                if re.match('[^#][a-zA-Z0-9.-]+:[0-9]+', line):
                    address, port = line.rsplit(':', 1)
                    servers.append((address, int(port)))
    if not servers:
        logger.info('No servers specified, defaulting to public.pyzor.org.')
        servers = [('public.pyzor.org', 24441)]
    return servers