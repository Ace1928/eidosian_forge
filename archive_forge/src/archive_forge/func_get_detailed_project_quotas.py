import abc
@staticmethod
@abc.abstractmethod
def get_detailed_project_quotas(context, resources, project_id):
    """Retrieve detailed quotas for the given list of resources and project

        :param context: The request context, for access checks.
        :param resources: A dictionary of the registered resource keys.
        :param project_id: The ID of the project to return quotas for.
        :return dict: mapping resource name in dict to its corresponding limit
                      used and reserved. Reserved currently returns default
                      value of 0
        """