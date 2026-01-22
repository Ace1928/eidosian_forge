Return details of the next batch in a batched update.

    The result is a tuple containing the new size of the group and the number
    of members that may receive the new definition (by a combination of
    creating new members and updating existing ones).

    Inputs are the target size for the group, the current size of the group,
    the number of members that already have the latest definition, the batch
    size, and the minimum number of members to keep in service during a rolling
    update.
    