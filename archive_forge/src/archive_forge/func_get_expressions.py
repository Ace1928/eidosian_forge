from boto.cloudsearch2.optionstatus import IndexFieldStatus
from boto.cloudsearch2.optionstatus import ServicePoliciesStatus
from boto.cloudsearch2.optionstatus import ExpressionStatus
from boto.cloudsearch2.optionstatus import AvailabilityOptionsStatus
from boto.cloudsearch2.optionstatus import ScalingParametersStatus
from boto.cloudsearch2.document import DocumentServiceConnection
from boto.cloudsearch2.search import SearchConnection
def get_expressions(self, names=None):
    """
        Return a list of rank expressions defined for this domain.
        :return: list of ExpressionStatus objects
        :rtype: list of :class:`boto.cloudsearch2.option.ExpressionStatus`
            object
        """
    fn = self.layer1.describe_expressions
    data = fn(self.name, names)
    data = data['DescribeExpressionsResponse']['DescribeExpressionsResult']['Expressions']
    return [ExpressionStatus(self, d, fn) for d in data]