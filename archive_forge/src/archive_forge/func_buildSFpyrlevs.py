import numpy as np
import scipy.misc as sc
import scipy.signal
import scipy.io
def buildSFpyrlevs(self, lodft, log_rad, angle, Xrcos, Yrcos, ht):
    if ht <= 1:
        lo0 = np.fft.ifft2(np.fft.ifftshift(lodft))
        coeff = [lo0.real]
    else:
        Xrcos = Xrcos - np.log2(2)
        himask = self.pointOp(log_rad, Yrcos, Xrcos)
        lutsize = 1024
        Xcosn = np.pi * np.array(range(-(2 * lutsize + 1), lutsize + 2)) / lutsize
        order = self.nbands - 1
        const = 2 ** (2 * order) * sc.factorial(order) ** 2 / (self.nbands * sc.factorial(2 * order))
        Ycosn = np.sqrt(const) * np.cos(Xcosn) ** order
        M, N = np.shape(lodft)
        orients = np.zeros((int(self.nbands), M, N))
        for b in range(int(self.nbands)):
            anglemask = self.pointOp(angle, Ycosn, Xcosn + np.pi * b / self.nbands).astype(np.complex)
            banddft = np.complex(0, -1) ** order * lodft
            banddft *= anglemask
            banddft *= himask
            orients[b, :, :] = np.fft.ifft2(np.fft.ifftshift(banddft)).real
        dims = np.array(lodft.shape)
        lostart = np.ceil((dims + 0.5) / 2) - np.ceil((np.ceil((dims - 0.5) / 2) + 0.5) / 2)
        loend = lostart + np.ceil((dims - 0.5) / 2)
        lostart = lostart.astype(int)
        loend = loend.astype(int)
        log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
        angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
        lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
        YIrcos = np.abs(np.sqrt(1 - Yrcos * Yrcos))
        lomask = self.pointOp(log_rad, YIrcos, Xrcos)
        lodft = lomask * lodft
        coeff = self.buildSFpyrlevs(lodft, log_rad, angle, Xrcos, Yrcos, ht - 1)
        coeff.insert(0, orients)
    return coeff