import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def describe_evaluations(self, filter_variable=None, eq=None, gt=None, lt=None, ge=None, le=None, ne=None, prefix=None, sort_order=None, next_token=None, limit=None):
    """
        Returns a list of `DescribeEvaluations` that match the search
        criteria in the request.

        :type filter_variable: string
        :param filter_variable:
        Use one of the following variable to filter a list of `Evaluation`
            objects:


        + `CreatedAt` - Sets the search criteria to the `Evaluation` creation
              date.
        + `Status` - Sets the search criteria to the `Evaluation` status.
        + `Name` - Sets the search criteria to the contents of `Evaluation` **
              ** `Name`.
        + `IAMUser` - Sets the search criteria to the user account that invoked
              an `Evaluation`.
        + `MLModelId` - Sets the search criteria to the `MLModel` that was
              evaluated.
        + `DataSourceId` - Sets the search criteria to the `DataSource` used in
              `Evaluation`.
        + `DataUri` - Sets the search criteria to the data file(s) used in
              `Evaluation`. The URL can identify either a file or an Amazon
              Simple Storage Solution (Amazon S3) bucket or directory.

        :type eq: string
        :param eq: The equal to operator. The `Evaluation` results will have
            `FilterVariable` values that exactly match the value specified with
            `EQ`.

        :type gt: string
        :param gt: The greater than operator. The `Evaluation` results will
            have `FilterVariable` values that are greater than the value
            specified with `GT`.

        :type lt: string
        :param lt: The less than operator. The `Evaluation` results will have
            `FilterVariable` values that are less than the value specified with
            `LT`.

        :type ge: string
        :param ge: The greater than or equal to operator. The `Evaluation`
            results will have `FilterVariable` values that are greater than or
            equal to the value specified with `GE`.

        :type le: string
        :param le: The less than or equal to operator. The `Evaluation` results
            will have `FilterVariable` values that are less than or equal to
            the value specified with `LE`.

        :type ne: string
        :param ne: The not equal to operator. The `Evaluation` results will
            have `FilterVariable` values not equal to the value specified with
            `NE`.

        :type prefix: string
        :param prefix:
        A string that is found at the beginning of a variable, such as `Name`
            or `Id`.

        For example, an `Evaluation` could have the `Name`
            `2014-09-09-HolidayGiftMailer`. To search for this `Evaluation`,
            select `Name` for the `FilterVariable` and any of the following
            strings for the `Prefix`:


        + 2014-09
        + 2014-09-09
        + 2014-09-09-Holiday

        :type sort_order: string
        :param sort_order: A two-value parameter that determines the sequence
            of the resulting list of `Evaluation`.

        + `asc` - Arranges the list in ascending order (A-Z, 0-9).
        + `dsc` - Arranges the list in descending order (Z-A, 9-0).


        Results are sorted by `FilterVariable`.

        :type next_token: string
        :param next_token: The ID of the page in the paginated results.

        :type limit: integer
        :param limit: The maximum number of `Evaluation` to include in the
            result.

        """
    params = {}
    if filter_variable is not None:
        params['FilterVariable'] = filter_variable
    if eq is not None:
        params['EQ'] = eq
    if gt is not None:
        params['GT'] = gt
    if lt is not None:
        params['LT'] = lt
    if ge is not None:
        params['GE'] = ge
    if le is not None:
        params['LE'] = le
    if ne is not None:
        params['NE'] = ne
    if prefix is not None:
        params['Prefix'] = prefix
    if sort_order is not None:
        params['SortOrder'] = sort_order
    if next_token is not None:
        params['NextToken'] = next_token
    if limit is not None:
        params['Limit'] = limit
    return self.make_request(action='DescribeEvaluations', body=json.dumps(params))