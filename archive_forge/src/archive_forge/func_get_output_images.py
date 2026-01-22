import logging
from typing import Any, Dict, List, Sequence
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from .utils import (
def get_output_images(self, response: Response) -> List:
    """Unpack the generated images, audio, video, etc. from the Diffusion Pipeline's response.

        Arguments:
            response: (wandb.sdk.integration_utils.auto_logging.Response) The response from
                the request.

        Returns:
            List of generated images, audio, video, etc.
        """
    if 'output-type' not in SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]:
        return response.images
    elif SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['output-type'] == 'video':
        if self.pipeline_name in ['ShapEPipeline']:
            return response.images
        return response.frames
    elif SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['output-type'] == 'audio':
        return response.audios