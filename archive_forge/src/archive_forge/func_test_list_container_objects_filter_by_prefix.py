import pytest
from libcloud.storage.drivers.dummy import DummyStorageDriver
def test_list_container_objects_filter_by_prefix(driver, container_with_contents):
    container_name, object_name = container_with_contents
    container = driver.get_container(container_name)
    objects = driver.list_container_objects(container=container, prefix=object_name[:3])
    assert any((o for o in objects if o.name == object_name))
    objects = driver.list_container_objects(container=container, prefix='does-not-exist.dat')
    assert not objects