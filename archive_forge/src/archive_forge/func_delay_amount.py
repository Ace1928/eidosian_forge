def delay_amount(self, context):
    """Calculate how long we should delay before retrying.

        :type context: RetryContext

        """
    raise NotImplementedError('delay_amount')