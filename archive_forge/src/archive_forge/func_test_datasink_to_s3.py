import os
import copy
import simplejson
import glob
import os.path as op
from subprocess import Popen
import hashlib
from collections import namedtuple
import pytest
import nipype
import nipype.interfaces.io as nio
from nipype.interfaces.base.traits_extension import isdefined
from nipype.interfaces.base import Undefined, TraitError
from nipype.utils.filemanip import dist_is_editable
from subprocess import check_call, CalledProcessError
@pytest.mark.skipif(noboto3 or not fakes3, reason='boto3 or fakes3 library is not available')
def test_datasink_to_s3(dummy_input, tmpdir):
    """
    This function tests to see if the S3 functionality of a DataSink
    works properly
    """
    ds = nio.DataSink()
    bucket_name = 'test'
    container = 'outputs'
    attr_folder = 'text_file'
    output_dir = 's3://' + bucket_name
    fakes3_dir = tmpdir.strpath
    input_path = dummy_input
    proc = Popen(['fakes3', '-r', fakes3_dir, '-p', '4567'], stdout=open(os.devnull, 'wb'))
    resource = boto3.resource(aws_access_key_id='mykey', aws_secret_access_key='mysecret', service_name='s3', endpoint_url='http://127.0.0.1:4567', use_ssl=False)
    resource.meta.client.meta.events.unregister('before-sign.s3', fix_s3_host)
    bucket = resource.create_bucket(Bucket=bucket_name)
    ds.inputs.base_directory = output_dir
    ds.inputs.container = container
    ds.inputs.bucket = bucket
    setattr(ds.inputs, attr_folder, input_path)
    ds.run()
    key = '/'.join([container, attr_folder, os.path.basename(input_path)])
    obj = bucket.Object(key=key)
    dst_md5 = obj.e_tag.replace('"', '')
    src_md5 = hashlib.md5(open(input_path, 'rb').read()).hexdigest()
    proc.kill()
    assert src_md5 == dst_md5