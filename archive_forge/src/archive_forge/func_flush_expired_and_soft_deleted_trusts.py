import abc
from keystone import exception
@abc.abstractmethod
def flush_expired_and_soft_deleted_trusts(self, project_id=None, trustor_user_id=None, trustee_user_id=None, date=None):
    """Flush expired and non-expired soft deleted trusts from the backend.

        :param project_id: ID of a project to filter trusts by.
        :param trustor_user_id: ID of a trustor_user_id to filter trusts by.
        :param trustee_user_id: ID of a trustee_user_id to filter trusts by.
        :param date: date to filter trusts by.
        :type date: datetime

        """
    raise exception.NotImplemented()