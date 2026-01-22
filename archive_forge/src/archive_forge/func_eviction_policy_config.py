import dill
from pydantic import BaseModel, Field, root_validator
from pydantic.types import ByteSize
from typing import Union, Any, Dict, Optional
from lazyops.configs.base import DefaultSettings
from lazyops.types import lazyproperty
from lazyops.libs.sqlcache.constants import DEFAULT_SETTINGS, OPTIMIZED_SETTINGS, DBNAME
@property
def eviction_policy_config(self):
    return get_eviction_policies(self.table_name)[self.eviction_policy]