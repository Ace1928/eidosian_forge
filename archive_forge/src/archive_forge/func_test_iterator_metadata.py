from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.commands.cp import DestinationInfo
from gslib.name_expansion import CopyObjectsIterator
from gslib.name_expansion import NameExpansionIteratorDestinationTuple
from gslib.name_expansion import NameExpansionResult
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
def test_iterator_metadata(self):
    src_strings_array = [['gs://bucket1'], ['source'], ['s3://bucket1']]
    dst_strings = ['gs://bucket2', 'dest', 'gs://bucket2']
    copy_objects_iterator = CopyObjectsIterator(_ConstrcutNameExpansionIteratorDestinationTupleIterator(src_strings_array, dst_strings), False)
    self.assertFalse(copy_objects_iterator.has_cloud_src)
    self.assertFalse(copy_objects_iterator.has_file_src)
    self.assertEqual(len(copy_objects_iterator.provider_types), 0)
    next(copy_objects_iterator)
    self.assertTrue(copy_objects_iterator.has_cloud_src)
    self.assertFalse(copy_objects_iterator.has_file_src)
    self.assertEqual(len(copy_objects_iterator.provider_types), 1)
    self.assertTrue('gs' in copy_objects_iterator.provider_types)
    next(copy_objects_iterator)
    self.assertTrue(copy_objects_iterator.has_cloud_src)
    self.assertTrue(copy_objects_iterator.has_file_src)
    self.assertEqual(len(copy_objects_iterator.provider_types), 2)
    self.assertTrue('file' in copy_objects_iterator.provider_types)
    self.assertFalse(copy_objects_iterator.is_daisy_chain)
    next(copy_objects_iterator)
    self.assertEqual(len(copy_objects_iterator.provider_types), 3)
    self.assertTrue('s3' in copy_objects_iterator.provider_types)
    self.assertTrue(copy_objects_iterator.is_daisy_chain)