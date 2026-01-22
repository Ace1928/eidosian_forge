import uuid
def generate_uuid(dashed=True):
    """Creates a random uuid string.

    :param dashed: Generate uuid with dashes or not
    :type dashed: bool
    :returns: string
    """
    if dashed:
        return str(uuid.uuid4())
    return uuid.uuid4().hex