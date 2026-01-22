import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def assign_policy(self, peer, policy, typ):
    if peer not in self.peers:
        raise Exception('peer {0} not found'.format(peer.name))
    name = policy['name']
    if name not in self.policies:
        raise Exception('policy {0} not found'.format(name))
    self.peers[peer]['policies'][typ] = policy