import decimal
import json
from datetime import datetime
from botocore.exceptions import ParamValidationError
from botocore.utils import is_json_value_header, parse_to_aware_datetime
def serialize_to_request(self, parameters, operation_model):
    input_shape = operation_model.input_shape
    if input_shape is not None:
        report = self._param_validator.validate(parameters, operation_model.input_shape)
        if report.has_errors():
            raise ParamValidationError(report=report.generate_report())
    return self._serializer.serialize_to_request(parameters, operation_model)