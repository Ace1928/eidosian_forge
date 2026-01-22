from unittest import mock
import uuid
from openstack.network.v2 import agent as _agent
from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from neutronclient.tests.unit.osc.v2 import fakes
@staticmethod
def create_bgp_speakers(attrs=None, count=1):
    """Create multiple fake bgp speakers.

        """
    bgp_speakers = []
    for i in range(count):
        bgp_speaker = FakeBgpSpeaker.create_one_bgp_speaker(attrs)
        bgp_speakers.append(bgp_speaker)
    return bgp_speakers