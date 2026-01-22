import inspect
import types
from botocore.docs.example import (
from botocore.docs.params import (
Documents an individual method

    :param section: The section to write to

    :param method_name: The name of the method

    :param operation_model: The model of the operation

    :param event_emitter: The event emitter to use to emit events

    :param example_prefix: The prefix to use in the method example.

    :type include_input: Dictionary where keys are parameter names and
        values are the shapes of the parameter names.
    :param include_input: The parameter shapes to include in the
        input documentation.

    :type include_output: Dictionary where keys are parameter names and
        values are the shapes of the parameter names.
    :param include_input: The parameter shapes to include in the
        output documentation.

    :type exclude_input: List of the names of the parameters to exclude.
    :param exclude_input: The names of the parameters to exclude from
        input documentation.

    :type exclude_output: List of the names of the parameters to exclude.
    :param exclude_input: The names of the parameters to exclude from
        output documentation.

    :param document_output: A boolean flag to indicate whether to
        document the output.

    :param include_signature: Whether or not to include the signature.
        It is useful for generating docstrings.
    