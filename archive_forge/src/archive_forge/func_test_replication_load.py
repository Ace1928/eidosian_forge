import collections
import http.client as http
import io
from unittest import mock
import copy
import os
import sys
import uuid
import fixtures
from oslo_serialization import jsonutils
import webob
from glance.cmd import replicator as glance_replicator
from glance.common import exception
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
def test_replication_load(self):
    tempdir = self.useFixture(fixtures.TempDir()).path

    def write_image(img, data):
        imgfile = os.path.join(tempdir, img['id'])
        with open(imgfile, 'w') as f:
            f.write(jsonutils.dumps(img))
        if data:
            with open('%s.img' % imgfile, 'w') as f:
                f.write(data)
    for img in FAKEIMAGES:
        cimg = copy.copy(img)
        if cimg['id'] == '5dcddce0-cba5-4f18-9cf4-9853c7b207a6':
            cimg['extra'] = 'thisissomeextra'
        if cimg['id'] == 'f4da1d2a-40e8-4710-b3aa-0222a4cc887b':
            cimg['dontrepl'] = 'thisisyetmoreextra'
        write_image(cimg, 'kjdhfkjshdfkjhsdkfd')
    new_id = str(uuid.uuid4())
    cimg['id'] = new_id
    write_image(cimg, 'dskjfhskjhfkfdhksjdhf')
    new_id_missing_data = str(uuid.uuid4())
    cimg['id'] = new_id_missing_data
    write_image(cimg, None)
    badfile = os.path.join(tempdir, 'kjdfhf')
    with open(badfile, 'w') as f:
        f.write(jsonutils.dumps([1, 2, 3, 4, 5]))
    options = collections.UserDict()
    options.dontreplicate = 'dontrepl dontreplabsent'
    options.targettoken = 'targettoken'
    args = ['localhost:9292', tempdir]
    orig_img_service = glance_replicator.get_image_service
    try:
        glance_replicator.get_image_service = get_image_service
        updated = glance_replicator.replication_load(options, args)
    finally:
        glance_replicator.get_image_service = orig_img_service
    self.assertIn('5dcddce0-cba5-4f18-9cf4-9853c7b207a6', updated)
    self.assertNotIn('f4da1d2a-40e8-4710-b3aa-0222a4cc887b', updated)
    self.assertIn(new_id, updated)
    self.assertNotIn(new_id_missing_data, updated)