import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def set_provision_state(self, session, target, config_drive=None, clean_steps=None, rescue_password=None, wait=False, timeout=None, deploy_steps=None, service_steps=None):
    """Run an action modifying this node's provision state.

        This call is asynchronous, it will return success as soon as the Bare
        Metal service acknowledges the request.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param target: Provisioning action, e.g. ``active``, ``provide``.
            See the Bare Metal service documentation for available actions.
        :param config_drive: Config drive to pass to the node, only valid
            for ``active` and ``rebuild`` targets. You can use functions from
            :mod:`openstack.baremetal.configdrive` to build it.
        :param clean_steps: Clean steps to execute, only valid for ``clean``
            target.
        :param rescue_password: Password for the rescue operation, only valid
            for ``rescue`` target.
        :param wait: Whether to wait for the target state to be reached.
        :param timeout: Timeout (in seconds) to wait for the target state to be
            reached. If ``None``, wait without timeout.
        :param deploy_steps: Deploy steps to execute, only valid for ``active``
            and ``rebuild`` target.
        :param service_steps: Service steps to execute, only valid for
            ``service`` target.

        :return: This :class:`Node` instance.
        :raises: ValueError if ``config_drive``, ``clean_steps``,
            ``deploy_steps`` or ``rescue_password`` are provided with an
            invalid ``target``.
        :raises: :class:`~openstack.exceptions.ResourceFailure` if the node
            reaches an error state while waiting for the state.
        :raises: :class:`~openstack.exceptions.ResourceTimeout` if timeout
            is reached while waiting for the state.
        """
    session = self._get_session(session)
    version = None
    if target in _common.PROVISIONING_VERSIONS:
        version = '1.%d' % _common.PROVISIONING_VERSIONS[target]
    if config_drive:
        if isinstance(config_drive, dict):
            version = _common.CONFIG_DRIVE_DICT_VERSION
        elif target == 'rebuild':
            version = _common.CONFIG_DRIVE_REBUILD_VERSION
    if deploy_steps:
        version = _common.DEPLOY_STEPS_VERSION
    version = self._assert_microversion_for(session, 'commit', version)
    body = {'target': target}
    if config_drive:
        if target not in ('active', 'rebuild'):
            raise ValueError('Config drive can only be provided with "active" and "rebuild" targets')
        body['configdrive'] = config_drive
    if clean_steps is not None:
        if target != 'clean':
            raise ValueError('Clean steps can only be provided with "clean" target')
        body['clean_steps'] = clean_steps
    if deploy_steps is not None:
        if target not in ('active', 'rebuild'):
            raise ValueError('Deploy steps can only be provided with "deploy" and "rebuild" target')
        body['deploy_steps'] = deploy_steps
    if service_steps is not None:
        if target != 'service':
            raise ValueError('Service steps can only be provided with "service" target')
        body['service_steps'] = service_steps
    if rescue_password is not None:
        if target != 'rescue':
            raise ValueError('Rescue password can only be provided with "rescue" target')
        body['rescue_password'] = rescue_password
    if wait:
        try:
            expected_state = _common.EXPECTED_STATES[target]
        except KeyError:
            raise ValueError('For target %s the expected state is not known, cannot wait for it' % target)
    request = self._prepare_request(requires_id=True)
    request.url = utils.urljoin(request.url, 'states', 'provision')
    response = session.put(request.url, json=body, headers=request.headers, microversion=version, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    msg = 'Failed to set provision state for bare metal node {node} to {target}'.format(node=self.id, target=target)
    exceptions.raise_from_response(response, error_message=msg)
    if wait:
        return self.wait_for_provision_state(session, expected_state, timeout=timeout)
    else:
        return self.fetch(session)