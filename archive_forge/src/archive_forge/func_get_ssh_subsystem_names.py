from ncclient.xml_ import BASE_NS_1_0
from ncclient.operations.third_party.nexus.rpc import ExecCommand
from .default import DefaultDeviceHandler
def get_ssh_subsystem_names(self):
    """
        Return a list of possible SSH subsystem names.

        Different NXOS versions use different SSH subsystem names for netconf.
        Therefore, we return a list so that several can be tried, if necessary.

        The Nexus device handler also accepts

        """
    preferred_ssh_subsystem = self.device_params.get('ssh_subsystem_name')
    name_list = ['netconf', 'xmlagent']
    if preferred_ssh_subsystem:
        return [preferred_ssh_subsystem] + [n for n in name_list if n != preferred_ssh_subsystem]
    else:
        return name_list