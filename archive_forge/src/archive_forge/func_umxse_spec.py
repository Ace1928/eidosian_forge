from openunmix import utils
import torch.hub
def umxse_spec(targets=None, device='cpu', pretrained=True):
    target_urls = {'speech': 'https://zenodo.org/records/3786908/files/speech_f5e0d9f9.pth', 'noise': 'https://zenodo.org/records/3786908/files/noise_04a6fc2d.pth'}
    from .model import OpenUnmix
    if targets is None:
        targets = ['speech', 'noise']
    max_bin = utils.bandwidth_to_max_bin(rate=16000.0, n_fft=1024, bandwidth=16000)
    target_models = {}
    for target in targets:
        target_unmix = OpenUnmix(nb_bins=1024 // 2 + 1, nb_channels=1, hidden_size=256, max_bin=max_bin)
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(target_urls[target], map_location=device)
            target_unmix.load_state_dict(state_dict, strict=False)
            target_unmix.eval()
        target_unmix.to(device)
        target_models[target] = target_unmix
    return target_models