# @Email: rah3156@rit.edu
# @Website: https://ryanhartzell.github.io/Bildkedde
# @License: https://github.com/RyanHartzell/Bildkedde/blob/master/LICENSE
# @github: https://github.com/RyanHartzell/Bildkedde
#
# Copyright (c) 2020 Ryan Hartzell
#
import time

import numpy as np
from scipy import constants
import cupy as cp

np.random.seed(69)
# cp.random.seed(69)

POISSON_APPROX_THRESH = 1000 # Rule of thumb, tunable

# Custom code for PSF/MTF and PRF approaches
# Can all be used in conjunction, but ideally I can make two approaches: 
#   1) traditional PSF + PRF routine
#   2) MTF-based routine, where I create a supersampled array per wavelength, and multiply MTF against Object-Optical Field to project to image plane

def psf():
    pass

def mtf():
    pass

def prf():
    # ingest psf array
    pass

def trapz(arr):
    pass

def sampling_gymnastics():
    pass

# Scene building helpers (projecting precise locations in space to FPA, discretization of sources in time linked to >2xcutoff frequency)
def scene2focalplane(scene, scene2boresight_quat):
    pass

def discretize():
    pass

def compute_pixel_ifovs():
    pass

def radiance2irradiance():
    # Camera equation? Initially just want to project single point source's radiance through optics to focal plane (single pixels?) or just turn it into an aperture referred irradiance.
    pass

def object2radiance():
    # don't technically need this, but would be fun to have a basic radiometry solver for simple meshes of materials (can probably outsource this)
    pass

# Below is based on the pyradi source, as well as this paper by Konnik and Welsh: https://arxiv.org/pdf/1412.4031.pdf

def photon_energy_per_wavelength(wl=550e-9, mode="J"):
    E_p = (constants.Planck * constants.speed_of_light) / wl
    if mode=="J":
        return E_p
    elif mode=="eV":
        return E_p / constants.electron_volt

def irrad2electrons(aperture_reffered_irrad, wl=550e-9, aperture=1.0, integration_time=1.0, transmission=1.0, qe=1.0, eod=1.0, shot=True):
    xp = cp.get_array_module(aperture_reffered_irrad)

    electrons = (aperture_reffered_irrad * aperture * integration_time * transmission) / photon_energy_per_wavelength(wl, mode="J")

    # EOD should only be used (not 1) when modelling single pixel detector to account for optical effects without needing PRF module
    if shot==True:
        # My changes here may have caused something to BREAK big time...
        mask = electrons > POISSON_APPROX_THRESH
        electrons[mask] = np.clip(xp.random.normal(electrons[mask], np.sqrt(electrons[mask])), 0.0, None)
        electrons[~mask] = xp.random.poisson(electrons[~mask])

    return xp.round(qe * eod * electrons)

def apply_photo_response_non_uniformity(electrons, sigma=0.01):
    xp = cp.get_array_module(electrons)
    """
    sigma is roughly a percentage of electrons (per array element). Will likely allow for a read-in format in absolute scaling units (mean of 1)
    """
    return xp.round(electrons + electrons * xp.random.normal(0, sigma, electrons.shape)) # equivalent to 'electrons * np.random.normal(1, sig_as_a_true_percentage)

def dark_current():
    # Have the necessary info in the paper for silicon, and could probably implement some other materials based on analytical funcs/LUTs out there.
    # Or could also just stick with explicit for now
    pass

def dark_signal(electrons, dark_current, integration_time=1.0):
    xp = cp.get_array_module(electrons)
    return xp.round(xp.random.poisson(xp.ones_like(electrons) * dark_current * integration_time))

def apply_dark_signal(electrons, dark_signal):
    return electrons + dark_signal

def apply_dark_fixed_pattern_noise(electrons, dark_signal, Dn=0.1):
    xp = cp.get_array_module(electrons)
    # Dn is between 0.1 and 0.4 as per the paper for most CCDs and CMOS devices
    return electrons + xp.round(dark_signal + dark_signal * xp.random.lognormal(0, dark_signal*Dn, dark_signal.shape))

# Nonlinearity should be able to be calculated with a deviation factor, but also we should support a full LUT approach, where you have an entry for every possible electron (0%-120% full well)
def electrons2voltage(electrons, ve=1e-6, vv=1.0, ve_nonlinearity=0.0, vv_nonlinearity=0.0, offset=0.0, read_noise_electrons_sigma=10.):
    xp = cp.get_array_module(electrons)
    return xp.random.normal(electrons, read_noise_electrons_sigma) * ve * vv + offset # offset is just a completely flat bias in VOLTAGE

def voltage2counts(voltage, vswing=1.0, bits=12, offset=1000, mode="round"):
    xp = cp.get_array_module(voltage)
    # vswing = vref_adc - vmin_adc
    # we'll just call vmin=0 for now, and vref represents maximum voltage which can be captured by ADC
    K = vswing / 2**bits
    if mode=="round":
        return xp.round(voltage / K).astype(xp.int64) + offset
    elif mode=="floor":
        return xp.floor(voltage / K).astype(xp.int64) + offset

if __name__ == "__main__":
    start = time.time()

    # # Example 1) Single Pixel Detector
    # irrad = 0.5e-14

    # # a) Convert incident irradiance to detected electrons
    # electrons = irrad2electrons(irrad)
    # print(f"# of electrons = {electrons}")

    # # b) Convert detected electron charge to voltage
    # voltage = electrons2voltage(electrons)
    # print(f"Measured voltage = {voltage}")

    # # c) Convert measured voltage to digital counts via ADC
    # counts = voltage2counts(voltage)

    # print(f"For a simple single pixel detector, and only shot noise, the counts [DN] returned for an input irradiance of {irrad} is {counts}.")

    # Example 2) Staring Detector Array
    # irrad = np.random.uniform(1e-15, 1e-13, 256*256).reshape((256,256))
    # irrad = np.zeros((256,256))
    # irrad = np.ones((256,256)) * 1e-12
    # irrad = np.ones((3,4096,4096)) * 1e-12


    # Test with GPU array as input
    irrad = cp.random.uniform(1e-15, 1e-13, 256*256).reshape((256,256))
    # irrad = cp.zeros((256,256))
    # irrad = cp.ones((256,256)) * 1e-12
    # irrad = cp.ones((3,4096,4096)) * 1e-12

    # Run memory check (here we would queue up a bunch of sensible chunks somehow...)
    # if irrad.nbytes >= cp.cuda.Device(cp.cuda.get_device_id()).mem_info[1]:
    #     raise MemoryError(f"This job must be batched, with a GPU limit of {cp.cuda.Device(cp.cuda.get_device_id()).mem_info[1]} and total input size of {irrad.nbytes}. Exiting...")

    print(f"Irrad: Mean={irrad.mean()}, Max={irrad.max()}, Min={irrad.min()}")

    # Same pipeline as before
    electrons = irrad2electrons(irrad)
    print(f"Electrons: Mean={electrons.mean()}, Max={electrons.max()}, Min={electrons.min()}")
    del irrad

    electrons = apply_photo_response_non_uniformity(electrons, 0.01)
    print(f"Post-PRNU: Mean={electrons.mean()}, Max={electrons.max()}, Min={electrons.min()}")

    electrons = apply_dark_fixed_pattern_noise(electrons, dark_signal(electrons, 10)) # electrons per second, and poisson process applied
    print(f"Post-Dark: Mean={electrons.mean()}, Max={electrons.max()}, Min={electrons.min()}")

    voltage = electrons2voltage(electrons)
    print(f"Voltage: Mean={voltage.mean()}, Max={voltage.max()}, Min={voltage.min()}")
    del electrons

    counts = voltage2counts(voltage)

    print(f"The staring detector array model results in an image of counts given an array of input irradiances.")
    print(f"Total execution time = {time.time() - start} [s]")
    print(f"Counts: Mean={counts.mean()}, Max={counts.max()}, Min={counts.min()}")

    if isinstance(counts, cp.ndarray):
        counts = counts.get()

    import matplotlib.pyplot as plt

    plt.imshow(counts, cmap="inferno")
    # fig, axes = plt.subplots(1,2)
    # axes[0].imshow(counts, cmap="inferno")
    # axes[1].plot(np.histogram(counts.flat, 1000)[0])
    plt.show()