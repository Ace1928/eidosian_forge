import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def create_data_source_from_s3(self, data_source_id, data_spec, data_source_name=None, compute_statistics=None):
    """
        Creates a `DataSource` object. A `DataSource` references data
        that can be used to perform CreateMLModel, CreateEvaluation,
        or CreateBatchPrediction operations.

        `CreateDataSourceFromS3` is an asynchronous operation. In
        response to `CreateDataSourceFromS3`, Amazon Machine Learning
        (Amazon ML) immediately returns and sets the `DataSource`
        status to `PENDING`. After the `DataSource` is created and
        ready for use, Amazon ML sets the `Status` parameter to
        `COMPLETED`. `DataSource` in `COMPLETED` or `PENDING` status
        can only be used to perform CreateMLModel, CreateEvaluation or
        CreateBatchPrediction operations.

        If Amazon ML cannot accept the input source, it sets the
        `Status` parameter to `FAILED` and includes an error message
        in the `Message` attribute of the GetDataSource operation
        response.

        The observation data used in a `DataSource` should be ready to
        use; that is, it should have a consistent structure, and
        missing data values should be kept to a minimum. The
        observation data must reside in one or more CSV files in an
        Amazon Simple Storage Service (Amazon S3) bucket, along with a
        schema that describes the data items by name and type. The
        same schema must be used for all of the data files referenced
        by the `DataSource`.

        After the `DataSource` has been created, it's ready to use in
        evaluations and batch predictions. If you plan to use the
        `DataSource` to train an `MLModel`, the `DataSource` requires
        another item: a recipe. A recipe describes the observation
        variables that participate in training an `MLModel`. A recipe
        describes how each input variable will be used in training.
        Will the variable be included or excluded from training? Will
        the variable be manipulated, for example, combined with
        another variable, or split apart into word combinations? The
        recipe provides answers to these questions. For more
        information, see the `Amazon Machine Learning Developer
        Guide`_.

        :type data_source_id: string
        :param data_source_id: A user-supplied identifier that uniquely
            identifies the `DataSource`.

        :type data_source_name: string
        :param data_source_name: A user-supplied name or description of the
            `DataSource`.

        :type data_spec: dict
        :param data_spec:
        The data specification of a `DataSource`:


        + DataLocationS3 - Amazon Simple Storage Service (Amazon S3) location
              of the observation data.
        + DataSchemaLocationS3 - Amazon S3 location of the `DataSchema`.
        + DataSchema - A JSON string representing the schema. This is not
              required if `DataSchemaUri` is specified.
        + DataRearrangement - A JSON string representing the splitting
              requirement of a `Datasource`. Sample - ` "{"randomSeed":"some-
              random-seed",
              "splitting":{"percentBegin":10,"percentEnd":60}}"`

        :type compute_statistics: boolean
        :param compute_statistics: The compute statistics for a `DataSource`.
            The statistics are generated from the observation data referenced
            by a `DataSource`. Amazon ML uses the statistics internally during
            an `MLModel` training. This parameter must be set to `True` if the
            ``DataSource `` needs to be used for `MLModel` training

        """
    params = {'DataSourceId': data_source_id, 'DataSpec': data_spec}
    if data_source_name is not None:
        params['DataSourceName'] = data_source_name
    if compute_statistics is not None:
        params['ComputeStatistics'] = compute_statistics
    return self.make_request(action='CreateDataSourceFromS3', body=json.dumps(params))