import abc
@staticmethod
@abc.abstractmethod
def get_all_quotas(context, resources):
    """Given a list of resources, retrieve the quotas for the all tenants.

        :param context: The request context, for access checks.
        :param resources: A dictionary of the registered resource keys.
        :return: quotas list of dict of project_id:, resourcekey1:
                 resourcekey2: ...
        """