from boto.cloudsearch2.optionstatus import IndexFieldStatus
from boto.cloudsearch2.optionstatus import ServicePoliciesStatus
from boto.cloudsearch2.optionstatus import ExpressionStatus
from boto.cloudsearch2.optionstatus import AvailabilityOptionsStatus
from boto.cloudsearch2.optionstatus import ScalingParametersStatus
from boto.cloudsearch2.document import DocumentServiceConnection
from boto.cloudsearch2.search import SearchConnection
def get_availability_options(self):
    """
        Return a :class:`boto.cloudsearch2.option.AvailabilityOptionsStatus`
        object representing the currently defined availability options for
        the domain.
        :return: OptionsStatus object
        :rtype: :class:`boto.cloudsearch2.option.AvailabilityOptionsStatus`
            object
        """
    return AvailabilityOptionsStatus(self, refresh_fn=self.layer1.describe_availability_options, refresh_key=['DescribeAvailabilityOptionsResponse', 'DescribeAvailabilityOptionsResult', 'AvailabilityOptions'], save_fn=self.layer1.update_availability_options)