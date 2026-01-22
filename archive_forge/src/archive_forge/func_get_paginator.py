import logging
from botocore import waiter, xform_name
from botocore.args import ClientArgsCreator
from botocore.auth import AUTH_TYPE_MAPS
from botocore.awsrequest import prepare_request_dict
from botocore.compress import maybe_compress_request
from botocore.config import Config
from botocore.credentials import RefreshableCredentials
from botocore.discovery import (
from botocore.docs.docstring import ClientMethodDocstring, PaginatorDocstring
from botocore.exceptions import (
from botocore.history import get_global_history_recorder
from botocore.hooks import first_non_none_response
from botocore.httpchecksum import (
from botocore.model import ServiceModel
from botocore.paginate import Paginator
from botocore.retries import adaptive, standard
from botocore.useragent import UserAgentString
from botocore.utils import (
from botocore.exceptions import ClientError  # noqa
from botocore.utils import S3ArnParamHandler  # noqa
from botocore.utils import S3ControlArnParamHandler  # noqa
from botocore.utils import S3ControlEndpointSetter  # noqa
from botocore.utils import S3EndpointSetter  # noqa
from botocore.utils import S3RegionRedirector  # noqa
from botocore import UNSIGNED  # noqa
def get_paginator(self, operation_name):
    """Create a paginator for an operation.

        :type operation_name: string
        :param operation_name: The operation name.  This is the same name
            as the method name on the client.  For example, if the
            method name is ``create_foo``, and you'd normally invoke the
            operation as ``client.create_foo(**kwargs)``, if the
            ``create_foo`` operation can be paginated, you can use the
            call ``client.get_paginator("create_foo")``.

        :raise OperationNotPageableError: Raised if the operation is not
            pageable.  You can use the ``client.can_paginate`` method to
            check if an operation is pageable.

        :rtype: ``botocore.paginate.Paginator``
        :return: A paginator object.

        """
    if not self.can_paginate(operation_name):
        raise OperationNotPageableError(operation_name=operation_name)
    else:
        actual_operation_name = self._PY_TO_OP_NAME[operation_name]

        def paginate(self, **kwargs):
            return Paginator.paginate(self, **kwargs)
        paginator_config = self._cache['page_config'][actual_operation_name]
        paginate.__doc__ = PaginatorDocstring(paginator_name=actual_operation_name, event_emitter=self.meta.events, service_model=self.meta.service_model, paginator_config=paginator_config, include_signature=False)
        service_module_name = get_service_module_name(self.meta.service_model)
        paginator_class_name = f'{service_module_name}.Paginator.{actual_operation_name}'
        documented_paginator_cls = type(paginator_class_name, (Paginator,), {'paginate': paginate})
        operation_model = self._service_model.operation_model(actual_operation_name)
        paginator = documented_paginator_cls(getattr(self, operation_name), paginator_config, operation_model)
        return paginator