@classmethod
def should_cycle(cls, datatype: str) -> bool:
    """
        Return whether we should cycle data based on the datatype.

        :param datatype:
            parlai datatype

        :return should_cycle:
            given datatype, return whether we should cycle
        """
    assert datatype is not None, 'datatype must not be none'
    return 'train' in datatype and 'evalmode' not in datatype and ('ordered' not in datatype)