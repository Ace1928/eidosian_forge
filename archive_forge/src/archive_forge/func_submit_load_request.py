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
def submit_load_request(self, image_id):
    img_path = self.image_path + '%012d.jpg' % image_id
    self.data_loader.request_load(self.receive_data, self.image_loader.load, (img_path,))