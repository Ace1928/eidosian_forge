import pytest
from libcloud.storage.drivers.dummy import DummyStorageDriver
@pytest.fixture
def container_with_contents(driver):
    container_name = 'test'
    object_name = 'test.dat'
    container = driver.create_container(container_name=container_name)
    driver.upload_object(__file__, container=container, object_name=object_name)
    return (container_name, object_name)