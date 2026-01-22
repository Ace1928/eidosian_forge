from __future__ import absolute_import, division, print_function
import copy
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
def pubnub_application(module, account):
    """Retrieve reference on target application from account model.

    NOTE: In case if account authorization will fail or there is no
    application with specified name, module will exit with error.
    :type module:   AnsibleModule
    :param module:  Reference on module which contain module launch
                    information and status report methods.
    :type account:  Account
    :param account: Reference on PubNub account model from which reference
                    on application should be fetched.

    :rtype:  Application
    :return: Reference on initialized and ready to use application model.
    """
    application = None
    params = module.params
    try:
        application = account.application(params['application'])
    except (exceptions.AccountError, exceptions.GeneralPubNubError) as exc:
        exc_msg = _failure_title_from_exception(exc)
        exc_descr = exc.message if hasattr(exc, 'message') else exc.args[0]
        module.fail_json(msg=exc_msg, description=exc_descr, changed=account.changed, module_cache=dict(account))
    if application is None:
        err_fmt = "There is no '{0}' application for {1}. Make sure what correct application name has been passed. If application doesn't exist you can create it on admin.pubnub.com."
        email = account.owner.email
        module.fail_json(msg=err_fmt.format(params['application'], email), changed=account.changed, module_cache=dict(account))
    return application