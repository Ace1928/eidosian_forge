import abc
@staticmethod
@abc.abstractmethod
def get_default_quotas(context, resources, project_id):
    """Retrieve the default quotas for the list of resources and project.

        :param context: The request context, for access checks.
        :param resources: A dictionary of the registered resource keys.
        :param project_id: The ID of the project to return default quotas for.
        :return: dict from resource name to dict of name and limit
        """