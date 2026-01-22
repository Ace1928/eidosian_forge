import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def create_ml_model(self, ml_model_id, ml_model_type, training_data_source_id, ml_model_name=None, parameters=None, recipe=None, recipe_uri=None):
    """
        Creates a new `MLModel` using the data files and the recipe as
        information sources.

        An `MLModel` is nearly immutable. Users can only update the
        `MLModelName` and the `ScoreThreshold` in an `MLModel` without
        creating a new `MLModel`.

        `CreateMLModel` is an asynchronous operation. In response to
        `CreateMLModel`, Amazon Machine Learning (Amazon ML)
        immediately returns and sets the `MLModel` status to
        `PENDING`. After the `MLModel` is created and ready for use,
        Amazon ML sets the status to `COMPLETED`.

        You can use the GetMLModel operation to check progress of the
        `MLModel` during the creation operation.

        CreateMLModel requires a `DataSource` with computed
        statistics, which can be created by setting
        `ComputeStatistics` to `True` in CreateDataSourceFromRDS,
        CreateDataSourceFromS3, or CreateDataSourceFromRedshift
        operations.

        :type ml_model_id: string
        :param ml_model_id: A user-supplied ID that uniquely identifies the
            `MLModel`.

        :type ml_model_name: string
        :param ml_model_name: A user-supplied name or description of the
            `MLModel`.

        :type ml_model_type: string
        :param ml_model_type: The category of supervised learning that this
            `MLModel` will address. Choose from the following types:

        + Choose `REGRESSION` if the `MLModel` will be used to predict a
              numeric value.
        + Choose `BINARY` if the `MLModel` result has two possible values.
        + Choose `MULTICLASS` if the `MLModel` result has a limited number of
              values.


        For more information, see the `Amazon Machine Learning Developer
            Guide`_.

        :type parameters: map
        :param parameters:
        A list of the training parameters in the `MLModel`. The list is
            implemented as a map of key/value pairs.

        The following is the current set of training parameters:


        + `sgd.l1RegularizationAmount` - Coefficient regularization L1 norm. It
              controls overfitting the data by penalizing large coefficients.
              This tends to drive coefficients to zero, resulting in sparse
              feature set. If you use this parameter, start by specifying a small
              value such as 1.0E-08. The value is a double that ranges from 0 to
              MAX_DOUBLE. The default is not to use L1 normalization. The
              parameter cannot be used when `L2` is specified. Use this parameter
              sparingly.
        + `sgd.l2RegularizationAmount` - Coefficient regularization L2 norm. It
              controls overfitting the data by penalizing large coefficients.
              This tends to drive coefficients to small, nonzero values. If you
              use this parameter, start by specifying a small value such as
              1.0E-08. The valuseis a double that ranges from 0 to MAX_DOUBLE.
              The default is not to use L2 normalization. This cannot be used
              when `L1` is specified. Use this parameter sparingly.
        + `sgd.maxPasses` - Number of times that the training process traverses
              the observations to build the `MLModel`. The value is an integer
              that ranges from 1 to 10000. The default value is 10.
        + `sgd.maxMLModelSizeInBytes` - Maximum allowed size of the model.
              Depending on the input data, the size of the model might affect its
              performance. The value is an integer that ranges from 100000 to
              2147483648. The default value is 33554432.

        :type training_data_source_id: string
        :param training_data_source_id: The `DataSource` that points to the
            training data.

        :type recipe: string
        :param recipe: The data recipe for creating `MLModel`. You must specify
            either the recipe or its URI. If you dont specify a recipe or its
            URI, Amazon ML creates a default.

        :type recipe_uri: string
        :param recipe_uri: The Amazon Simple Storage Service (Amazon S3)
            location and file name that contains the `MLModel` recipe. You must
            specify either the recipe or its URI. If you dont specify a recipe
            or its URI, Amazon ML creates a default.

        """
    params = {'MLModelId': ml_model_id, 'MLModelType': ml_model_type, 'TrainingDataSourceId': training_data_source_id}
    if ml_model_name is not None:
        params['MLModelName'] = ml_model_name
    if parameters is not None:
        params['Parameters'] = parameters
    if recipe is not None:
        params['Recipe'] = recipe
    if recipe_uri is not None:
        params['RecipeUri'] = recipe_uri
    return self.make_request(action='CreateMLModel', body=json.dumps(params))