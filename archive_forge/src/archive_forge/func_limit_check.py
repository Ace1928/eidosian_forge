import abc
@staticmethod
@abc.abstractmethod
def limit_check(context, project_id, resources, values):
    """Check simple quota limits.

        For limits--those quotas for which there is no usage
        synchronization function--this method checks that a set of
        proposed values are permitted by the limit restriction.

        If any of the proposed values is over the defined quota, an
        OverQuota exception will be raised with the sorted list of the
        resources which are too high.  Otherwise, the method returns
        nothing.

        :param context: The request context, for access checks.
        :param project_id: The ID of the project to make the reservations for.
        :param resources: A dictionary of the registered resource.
        :param values: A dictionary of the values to check against the
                       quota.
        """