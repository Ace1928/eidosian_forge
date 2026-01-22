from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def do_resolve(x):
    return x