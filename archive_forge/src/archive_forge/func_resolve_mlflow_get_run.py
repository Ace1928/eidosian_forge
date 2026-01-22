import graphene
import mlflow
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.utils.proto_json_utils import parse_dict
def resolve_mlflow_get_run(self, info, input):
    input_dict = vars(input)
    request_message = mlflow.protos.service_pb2.GetRun()
    parse_dict(input_dict, request_message)
    return mlflow.server.handlers.get_run_impl(request_message)