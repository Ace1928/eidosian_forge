import json
import os
import re
import numpy as np
from collections import Counter
from parlai.core.agents import Agent
from collections import defaultdict
from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from .build import build
from parlai.tasks.coco_caption.build_2014 import buildImage as buildImage_2014
from parlai.tasks.coco_caption.build_2015 import buildImage as buildImage_2015
def next_example(self):
    """
        Returns the next example from this dataset after starting to queue up the next
        example.
        """
    ready = None
    if self.example is not None:
        if self.image_mode != 'no_image_model' and 'image_id' in self.example:
            image = self.data_queue.get()
            self.example['image'] = image
        ready = (self.example, self.imageEpochDone)
    self.example, self.imageEpochDone = super().next_example()
    if self.image_mode != 'no_image_model' and 'image_id' in self.example:
        image_id = self.example['image_id']
        self.submit_load_request(image_id)
    if ready is None:
        return self.next_example()
    else:
        return ready