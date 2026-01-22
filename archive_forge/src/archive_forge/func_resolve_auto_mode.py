import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def resolve_auto_mode(self, region_name):
    current_region = None
    if os.environ.get('AWS_EXECUTION_ENV'):
        default_region = os.environ.get('AWS_DEFAULT_REGION')
        current_region = os.environ.get('AWS_REGION', default_region)
    if not current_region:
        if self._instance_metadata_region:
            current_region = self._instance_metadata_region
        else:
            try:
                current_region = self._imds_region_provider.provide()
                self._instance_metadata_region = current_region
            except Exception:
                pass
    if current_region:
        if region_name == current_region:
            return 'in-region'
        else:
            return 'cross-region'
    return 'standard'