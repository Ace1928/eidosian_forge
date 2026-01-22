import graphene
import mlflow
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.utils.proto_json_utils import parse_dict
def resolve_mlflow_get_experiment(self, info, input):
    input_dict = vars(input)
    request_message = mlflow.protos.service_pb2.GetExperiment()
    parse_dict(input_dict, request_message)
    return mlflow.server.handlers.get_experiment_impl(request_message)