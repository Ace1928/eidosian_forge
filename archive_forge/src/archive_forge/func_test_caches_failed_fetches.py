from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def test_caches_failed_fetches():

    @Promise.safe
    def do():
        identity_loader, load_calls = id_loader()
        identity_loader.prime(1, Exception('Error: 1'))
        with raises(Exception) as exc_info:
            identity_loader.load(1).get()
        assert load_calls == []
    do().get()