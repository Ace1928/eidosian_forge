from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def test_dataloader_clear_with_missing_key_works():

    @Promise.safe
    def do():

        def do_resolve(x):
            return x
        a_loader, a_load_calls = id_loader(resolve=do_resolve)
        assert a_loader.clear('A1') == a_loader
    do().get()