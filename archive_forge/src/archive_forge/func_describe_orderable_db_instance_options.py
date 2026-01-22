import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def describe_orderable_db_instance_options(self, engine, engine_version=None, db_instance_class=None, license_model=None, vpc=None, max_records=None, marker=None):
    """
        Returns a list of orderable DB instance options for the
        specified engine.

        :type engine: string
        :param engine: The name of the engine to retrieve DB instance options
            for.

        :type engine_version: string
        :param engine_version: The engine version filter value. Specify this
            parameter to show only the available offerings matching the
            specified engine version.

        :type db_instance_class: string
        :param db_instance_class: The DB instance class filter value. Specify
            this parameter to show only the available offerings matching the
            specified DB instance class.

        :type license_model: string
        :param license_model: The license model filter value. Specify this
            parameter to show only the available offerings matching the
            specified license model.

        :type vpc: boolean
        :param vpc: The VPC filter value. Specify this parameter to show only
            the available VPC or non-VPC offerings.

        :type max_records: integer
        :param max_records: The maximum number of records to include in the
            response. If more records exist than the specified `MaxRecords`
            value, a pagination token called a marker is included in the
            response so that the remaining results can be retrieved.
        Default: 100

        Constraints: minimum 20, maximum 100

        :type marker: string
        :param marker: An optional pagination token provided by a previous
            DescribeOrderableDBInstanceOptions request. If this parameter is
            specified, the response includes only records beyond the marker, up
            to the value specified by `MaxRecords` .

        """
    params = {'Engine': engine}
    if engine_version is not None:
        params['EngineVersion'] = engine_version
    if db_instance_class is not None:
        params['DBInstanceClass'] = db_instance_class
    if license_model is not None:
        params['LicenseModel'] = license_model
    if vpc is not None:
        params['Vpc'] = str(vpc).lower()
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeOrderableDBInstanceOptions', verb='POST', path='/', params=params)