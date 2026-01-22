import boto3
import os
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound
def create_hit_config(opt, task_description, unique_worker, is_sandbox):
    """
    Writes a HIT config to file.
    """
    mturk_submit_url = 'https://workersandbox.mturk.com/mturk/externalSubmit'
    if not is_sandbox:
        mturk_submit_url = 'https://www.mturk.com/mturk/externalSubmit'
    hit_config = {'task_description': task_description, 'is_sandbox': is_sandbox, 'mturk_submit_url': mturk_submit_url, 'unique_worker': unique_worker, 'frame_height': opt.get('frame_height', 650), 'allow_reviews': opt.get('allow_reviews', False), 'block_mobile': opt.get('block_mobile', True), 'chat_title': opt.get('chat_title', opt.get('hit_title', 'Live Chat')), 'template_type': opt.get('frontend_template_type', 'default')}
    hit_config_file_path = os.path.join(opt['tmp_dir'], 'hit_config.json')
    if os.path.exists(hit_config_file_path):
        os.remove(hit_config_file_path)
    with open(hit_config_file_path, 'w') as hit_config_file:
        hit_config_file.write(json.dumps(hit_config))