from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import time
from typing import List, Optional, Tuple
from absl import flags
from clients import bigquery_client
from clients import client_dataset
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import utils as frontend_utils
from utils import bq_error
from utils import bq_id_utils
def shouldContinueAfterCrossRegionCheck(self, client: bigquery_client.BigqueryClient, source_references: List[bq_id_utils.ApiClientHelper.TableReference], source_references_str: str, dest_reference: bq_id_utils.ApiClientHelper.TableReference, destination_region: str) -> bool:
    """Checks if it is a Cross Region Copy operation and obtains confirmation.

    Args:
      client: Bigquery client
      source_references: Source reference
      source_references_str: Source reference string
      dest_reference: Destination dataset reference
      destination_region: Destination dataset region

    Returns:
      true  - it is not a cross-region operation, or user has used force option,
              or cross-region operation is verified confirmed with user, or
              Insufficient permissions to query datasets for validation
      false - if user did not allow cross-region operation, or
              Dataset does not exist hence operation can't be performed.
    Raises:
      bq_error.BigqueryNotFoundError: If unable to compute the dataset
        region
    """
    destination_dataset = dest_reference.GetDatasetReference()
    try:
        all_source_datasets_in_same_region, first_source_region = self._CheckAllSourceDatasetsInSameRegionAndGetFirstSourceRegion(client, source_references)
        if destination_region is None:
            destination_region = client.GetDatasetRegion(destination_dataset)
    except bq_error.BigqueryAccessDeniedError as err:
        print('Unable to determine source or destination dataset location, skipping cross-region validation: ' + str(err))
        return True
    if destination_region is None:
        raise bq_error.BigqueryNotFoundError(self._DATASET_NOT_FOUND % (str(destination_dataset),), {'reason': 'notFound'}, [])
    if all_source_datasets_in_same_region and destination_region == first_source_region:
        return True
    print(self._NOTE, '\n' + self._SYNC_FLAG_ENABLED_WARNING if FLAGS.sync else '\n' + self._CROSS_REGION_WARNING)
    if self.force:
        return True
    if 'y' != frontend_utils.PromptYN(self._CONFIRM_CROSS_REGION % (source_references_str,)):
        print(self._NOT_COPYING % (source_references_str,))
        return False
    return True