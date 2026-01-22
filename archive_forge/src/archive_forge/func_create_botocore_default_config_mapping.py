import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def create_botocore_default_config_mapping(session):
    chain_builder = ConfigChainFactory(session=session)
    config_mapping = _create_config_chain_mapping(chain_builder, BOTOCORE_DEFAUT_SESSION_VARIABLES)
    config_mapping['s3'] = SectionConfigProvider('s3', session, _create_config_chain_mapping(chain_builder, DEFAULT_S3_CONFIG_VARS))
    config_mapping['proxies_config'] = SectionConfigProvider('proxies_config', session, _create_config_chain_mapping(chain_builder, DEFAULT_PROXIES_CONFIG_VARS))
    return config_mapping