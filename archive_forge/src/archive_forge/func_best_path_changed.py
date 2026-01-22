from os_ken.services.protocols.bgp.signals import SignalBus
def best_path_changed(self, path, is_withdraw):
    return self.emit_signal(self.BGP_BEST_PATH_CHANGED, {'path': path, 'is_withdraw': is_withdraw})