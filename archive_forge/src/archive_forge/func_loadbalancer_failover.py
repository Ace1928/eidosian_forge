from octavia_lib.api.drivers import exceptions
def loadbalancer_failover(self, loadbalancer_id):
    """Performs a fail over of a load balancer.

        :param loadbalancer_id: ID of the load balancer to failover.
        :type loadbalancer_id: string
        :return: Nothing if the failover request was accepted.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises: NotImplementedError if driver does not support request.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support failing over load balancers.', operator_fault_string='This provider does not support failing over load balancers.')