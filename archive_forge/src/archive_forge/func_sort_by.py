from typing import List, Union
def sort_by(self, *fields: List[str], **kwargs) -> 'AggregateRequest':
    """
        Indicate how the results should be sorted. This can also be used for
        *top-N* style queries

        ### Parameters

        - **fields**: The fields by which to sort. This can be either a single
            field or a list of fields. If you wish to specify order, you can
            use the `Asc` or `Desc` wrapper classes.
        - **max**: Maximum number of results to return. This can be
            used instead of `LIMIT` and is also faster.


        Example of sorting by `foo` ascending and `bar` descending:

        ```
        sort_by(Asc("@foo"), Desc("@bar"))
        ```

        Return the top 10 customers:

        ```
        AggregateRequest()            .group_by("@customer", r.sum("@paid").alias(FIELDNAME))            .sort_by(Desc("@paid"), max=10)
        ```
        """
    if isinstance(fields, (str, SortDirection)):
        fields = [fields]
    fields_args = []
    for f in fields:
        if isinstance(f, SortDirection):
            fields_args += [f.field, f.DIRSTRING]
        else:
            fields_args += [f]
    ret = ['SORTBY', str(len(fields_args))]
    ret.extend(fields_args)
    max = kwargs.get('max', 0)
    if max > 0:
        ret += ['MAX', str(max)]
    self._aggregateplan.extend(ret)
    return self