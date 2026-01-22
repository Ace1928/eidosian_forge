from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.container.gkemulticloud import errors
def verify_install_agent_deployed(kube_client: kube_util.KubernetesClient):
    """Verifies the install agent is deployed and running on the target cluster.

  Accesses the cluster and checks if the install agent is running to ensure
  subsequent operations can succeed. Raises MissingAttachedInstallAgent if the
  agent is not running or it can't be determined.

  Args:
    kube_client: Client to the cluster.
  """
    ns = kube_client.NamespacesWithLabelSelector('{}={}'.format('kubernetes.io/metadata.name', constants.ATTACHED_INSTALL_AGENT_NAMESPACE))
    if constants.ATTACHED_INSTALL_AGENT_NAMESPACE not in ns:
        raise errors.MissingAttachedInstallAgent('"{}" namespace is missing.'.format(constants.ATTACHED_INSTALL_AGENT_NAMESPACE))
    pod_name, err = kube_client.GetResourceField(constants.ATTACHED_INSTALL_AGENT_NAMESPACE, 'pods', '.items[*].metadata.name')
    if err is not None:
        raise errors.MissingAttachedInstallAgent('Unable to get pods: ' + err)
    status, err = kube_client.GetResourceField(constants.ATTACHED_INSTALL_AGENT_NAMESPACE, 'pods/{}'.format(pod_name), '.status.phase')
    if err is not None:
        raise errors.MissingAttachedInstallAgent('Unable to get install agent pod: ' + err)
    if status != 'Running':
        raise errors.MissingAttachedInstallAgent('Invalid install agent status: ' + status)