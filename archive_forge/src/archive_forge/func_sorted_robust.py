def sorted_robust(iterable, key=None, reverse=False):
    """Utility to sort an arbitrary iterable.

    This returns the sorted(arg) in a consistent order by first trying
    the standard sort() function, and if that fails (for example with
    mixed-type Sets in Python3), use the _robust_sort_keyfcn utility
    (above) to generate sortable keys.

    Parameters
    ----------
    iterable: iterable
        the source of items to sort
    key: function
        a function of one argument that is used to extract the
        comparison key from each element in `iterable`
    reverse: bool
        if True, the iterable is sorted as if each comparison was reversed.

    Returns
    -------
    list
    """
    ans = list(iterable)
    try:
        ans.sort(key=key, reverse=reverse)
    except:
        ans.sort(key=_robust_sort_keyfcn(key), reverse=reverse)
    return ans