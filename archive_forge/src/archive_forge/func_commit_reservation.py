import abc
@staticmethod
@abc.abstractmethod
def commit_reservation(context, reservation_id):
    """Commit a reservation register

        :param context: The request context, for access checks.
        :param reservation_id: ID of the reservation register to commit.
        """