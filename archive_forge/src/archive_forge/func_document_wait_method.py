import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.method import document_model_driven_method
from botocore.docs.utils import DocumentedShape
from botocore.utils import get_service_module_name
def document_wait_method(section, waiter_name, event_emitter, service_model, service_waiter_model, include_signature=True):
    """Documents a the wait method of a waiter

    :param section: The section to write to

    :param waiter_name: The name of the waiter

    :param event_emitter: The event emitter to use to emit events

    :param service_model: The service model

    :param service_waiter_model: The waiter model associated to the service

    :param include_signature: Whether or not to include the signature.
        It is useful for generating docstrings.
    """
    waiter_model = service_waiter_model.get_waiter(waiter_name)
    operation_model = service_model.operation_model(waiter_model.operation)
    waiter_config_members = OrderedDict()
    waiter_config_members['Delay'] = DocumentedShape(name='Delay', type_name='integer', documentation='<p>The amount of time in seconds to wait between attempts. Default: {}</p>'.format(waiter_model.delay))
    waiter_config_members['MaxAttempts'] = DocumentedShape(name='MaxAttempts', type_name='integer', documentation='<p>The maximum number of attempts to be made. Default: {}</p>'.format(waiter_model.max_attempts))
    botocore_waiter_params = [DocumentedShape(name='WaiterConfig', type_name='structure', documentation='<p>A dictionary that provides parameters to control waiting behavior.</p>', members=waiter_config_members)]
    wait_description = 'Polls :py:meth:`{}.Client.{}` every {} seconds until a successful state is reached. An error is returned after {} failed checks.'.format(get_service_module_name(service_model), xform_name(waiter_model.operation), waiter_model.delay, waiter_model.max_attempts)
    document_model_driven_method(section, 'wait', operation_model, event_emitter=event_emitter, method_description=wait_description, example_prefix='waiter.wait', include_input=botocore_waiter_params, document_output=False, include_signature=include_signature)