from typing import Any, Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.utils.input import get_color_mapping
from langchain.chains.base import Chain
@root_validator()
def validate_chains(cls, values: Dict) -> Dict:
    """Validate that chains are all single input/output."""
    for chain in values['chains']:
        if len(chain.input_keys) != 1:
            raise ValueError(f'Chains used in SimplePipeline should all have one input, got {chain} with {len(chain.input_keys)} inputs.')
        if len(chain.output_keys) != 1:
            raise ValueError(f'Chains used in SimplePipeline should all have one output, got {chain} with {len(chain.output_keys)} outputs.')
    return values