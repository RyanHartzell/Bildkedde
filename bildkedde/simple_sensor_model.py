import numpy as np
from scipy import constants

# Based on this pyradi source, as well as this paper by Konnik and Welsh: https://arxiv.org/pdf/1412.4031.pdf

def photon_energy_per_wavelength(l=550e-9, mode="J"):
    E_p = (constants.Planck * constants.speed_of_light) / l
    if mode=="J":
        return E_p
    elif mode=="eV":
        return E_p / constants.electron_volt

def irrad2electrons(irrad, l=550e-9, aperture=1.0, integration_time=1.0, transmission=1.0, qe=1.0, eod=1.0):
    return np.random.poisson((irrad * aperture * integration_time * transmission * qe * eod) / photon_energy_per_wavelength(l, mode="J"))

def electrons2voltage(electrons, ve=1e-6, vv=1.0):
    return electrons * ve * vv

def voltage2counts(voltage, vswing=1.0, bits=12, mode="round"):
    # vswing = vref_adc - vmin_adc
    # we'll just call vmin=0 for now, and vref represents maximum voltage which can be captured by ADC
    K = vswing / 2**bits
    if mode=="round":
        return np.round(voltage / K)
    elif mode=="floor":
        return np.floor(voltage / K)

if __name__ == "__main__":
    # Example 1) Single Pixel Detector
    irrad = 0.5e-14

    # a) Convert incident irradiance to detected electrons
    electrons = irrad2electrons(irrad)
    print(f"# of electrons = {electrons}")

    # b) Convert detected electron charge to voltage
    voltage = electrons2voltage(electrons)
    print(f"Measured voltage = {voltage}")

    # c) Convert measured voltage to digital counts via ADC
    counts = voltage2counts(voltage)

    print(f"For a simple single pixel detector, and only shot noise, the counts [DN] returned for an input irradiance of {irrad} is {counts}.")

    # Example 2) Staring Detector Array
    irrad = np.random.uniform(1e-17, 1e-15, 2048*2048).reshape((2048,2048))

    # Same pipeline as before
    electrons = irrad2electrons(irrad)
    voltage = electrons2voltage(electrons)
    counts = voltage2counts(voltage)

    print(f"The staring detector array model results in an image of counts given an array of input irradiances.")

    import matplotlib.pyplot as plt

    plt.imshow(counts, cmap="inferno")
    plt.show()