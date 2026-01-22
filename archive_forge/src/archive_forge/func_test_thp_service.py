import os
import testinfra.utils.ansible_runner
def test_thp_service(host):
    """
    Validates the service actually works
    """
    switches = ['/sys/kernel/mm/transparent_hugepage/enabled', '/sys/kernel/mm/transparent_hugepage/defrag']
    facts = host.ansible('setup')['ansible_facts']
    virt_types = facts.get('ansible_virtualization_tech_guest', [facts['ansible_virtualization_type']])
    in_docker = 'container' in virt_types or 'docker' in virt_types
    if not in_docker and facts['ansible_virtualization_role'] == 'guest':
        in_docker = host.file('/.dockerenv').exists
    if not in_docker:
        for d in switches:
            cmd = host.run('cat {0}'.format(d))
            assert cmd.rc == 0
            assert '[never]' in cmd.stdout