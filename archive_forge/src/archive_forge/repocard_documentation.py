import os
import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type, Union
import requests
import yaml
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import upload_file
from huggingface_hub.repocard_data import (
from huggingface_hub.utils import get_session, is_jinja_available, yaml_dump
from .constants import REPOCARD_NAME
from .utils import EntryNotFoundError, SoftTemporaryDirectory, logging, validate_hf_hub_args
Initialize a DatasetCard from a template. By default, it uses the default template, which can be found here:
        https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md

        Templates are Jinja2 templates that can be customized by passing keyword arguments.

        Args:
            card_data (`huggingface_hub.DatasetCardData`):
                A huggingface_hub.DatasetCardData instance containing the metadata you want to include in the YAML
                header of the dataset card on the Hugging Face Hub.
            template_path (`str`, *optional*):
                A path to a markdown file with optional Jinja template variables that can be filled
                in with `template_kwargs`. Defaults to the default template.

        Returns:
            [`huggingface_hub.DatasetCard`]: A DatasetCard instance with the specified card data and content from the
            template.

        Example:
            ```python
            >>> from huggingface_hub import DatasetCard, DatasetCardData

            >>> # Using the Default Template
            >>> card_data = DatasetCardData(
            ...     language='en',
            ...     license='mit',
            ...     annotations_creators='crowdsourced',
            ...     task_categories=['text-classification'],
            ...     task_ids=['sentiment-classification', 'text-scoring'],
            ...     multilinguality='monolingual',
            ...     pretty_name='My Text Classification Dataset',
            ... )
            >>> card = DatasetCard.from_template(
            ...     card_data,
            ...     pretty_name=card_data.pretty_name,
            ... )

            >>> # Using a Custom Template
            >>> card_data = DatasetCardData(
            ...     language='en',
            ...     license='mit',
            ... )
            >>> card = DatasetCard.from_template(
            ...     card_data=card_data,
            ...     template_path='./src/huggingface_hub/templates/datasetcard_template.md',
            ...     custom_template_var='custom value',  # will be replaced in template if it exists
            ... )

            ```
        