from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def flatten_aggregated_list(field_name: str):
    """Collect resources returned from an aggregated list response into a list.

  The aggregated list returns every instance of the resource in every scope
  in a particular form. A partial example is given below:
    {
      "kind": "compute#storagePoolAggregatedList",
      "id": "projects/cengizk-staging/aggregated/storagePools",
      "items": {
        "global": {
          "warning": {
            "code": "NO_RESULTS_ON_PAGE",
            "message": "There are no results for scope 'global' on this page.",
            "data": [
              {
                "key": "scope",
                "value": "global"
              }
            ]
          }
        },
        "zones/us-central1-ir1": {
          "storagePools": [
            {
              "kind": "compute#storagePool",
              "id": "2014221622075011414",
              "creationTimestamp": "2023-03-23T10:43:21.767-07:00",
              "name": "foo",
            },
            {
              "kind": "compute#storagePool",
              "id": "5926595245683775672",
              "creationTimestamp": "2023-04-05T09:48:24.926-07:00",
              "name": "foo2",
              "description": "",
              "sizeGb": "1",
              "provisionedIops": "5",
              "zone": "https://www.googleapis.com/compute/staging_alpha/
                projects/cengizk-staging/zones/us-central1-ir1",
              "state": "READY",
              "type": "SSD",
              "resourceStatus": {
                "numberOfDisks": "0",
                "aggregateDiskSizeGb": "0",
                "maxAggregateDiskSizeGb": "5",
                "usedBytes": "0",
                "usedReducedBytes": "0",
                "aggregateDiskProvisionedIops": "0",
                "lastResizeTimestamp": "1969-12-31T16:00:00.000-08:00"
              }
            }

  Args:
    field_name:  Name of the field that the resource is collected into. For
    the example above, for the zone "us-central1-ir1", the resources are
    collected into "storagePools", thus that must be the value. In general,
    this value should be the plural camelcase form of the resource type.

  Returns:
    A function that can serve as a response modification hook. See
    `modify_response_hooks` and `python_hook` in `yaml_command_schema.yaml`.
  """

    def hook(response, _):
        resources = []
        for scope in response.items.additionalProperties:
            resources += getattr(scope.value, field_name)
        return resources
    return hook