import base64
import io
from typing import Dict, Union
import PIL
import torch
from parlai.core.image_featurizers import ImageLoader
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.typing import TShared
def offload_state(self) -> Dict[str, str]:
    """
        Return serialized state.

        :return state_dict:
            serialized state that can be used in json.dumps
        """
    byte_arr = io.BytesIO()
    image = self.get_image()
    image.save(byte_arr, format='JPEG')
    serialized = base64.encodebytes(byte_arr.getvalue()).decode('utf-8')
    return {'image_id': self.get_image_id(), 'image_location_id': self.get_image_location_id(), 'image': serialized}