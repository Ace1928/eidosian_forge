from boto.cloudsearch2.optionstatus import IndexFieldStatus
from boto.cloudsearch2.optionstatus import ServicePoliciesStatus
from boto.cloudsearch2.optionstatus import ExpressionStatus
from boto.cloudsearch2.optionstatus import AvailabilityOptionsStatus
from boto.cloudsearch2.optionstatus import ScalingParametersStatus
from boto.cloudsearch2.document import DocumentServiceConnection
from boto.cloudsearch2.search import SearchConnection
def get_scaling_options(self):
    """
        Return a :class:`boto.cloudsearch2.option.ScalingParametersStatus`
        object representing the currently defined scaling options for the
        domain.
        :return: ScalingParametersStatus object
        :rtype: :class:`boto.cloudsearch2.option.ScalingParametersStatus`
            object
        """
    return ScalingParametersStatus(self, refresh_fn=self.layer1.describe_scaling_parameters, refresh_key=['DescribeScalingParametersResponse', 'DescribeScalingParametersResult', 'ScalingParameters'], save_fn=self.layer1.update_scaling_parameters)