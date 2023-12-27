import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve, convolve1d

import cupy as cp

np.random.seed(69)
# cp.random.seed(69)

# Start with basic convolution kernel in subpixel space, then expand to MTF method. Need higher subsampling rate and padding cleverness
# Think "filter banks"
# This implementation is ripped from image pypelines!!! I should think about integrating with it properly
def low_pass(img, cut_off, filter_type='ideal', butterworth_order=1):
    """calculates a lowpass filter for an input image

    Args:
        img(np.ndarray): image to calculate filter for
        cut_off (float): cutoff frequency for this filter. units in #TODO
        filter_type (str): the type of filter to apply, 'ideal','gaussian',
            'butterworth'
        butterworth_order(float): butterworth order if butterworth filter is
            being used

    Returns:
        filter(np.ndarray) 2D filter
    """
    # for now, assuming 2D array as input
    r,c = img.shape
    u = np.arange(r)
    v = np.arange(c)
    u, v = np.meshgrid(u, v)
    low_pass = np.sqrt( (u-r/2)**2 + (v-c/2)**2 )

    if filter_type == 'ideal':
        low_pass[low_pass <= cut_off] = 1
        low_pass[low_pass >= cut_off] = 0

    elif filter_type == 'gaussian':
        xp = -1*(low_pass**2) / (2* cut_off**2)
        low_pass = np.exp( xp )
        low_pass = np.clip(low_pass,0,1)

    elif filter_type == 'butterworth':
        denom = 1.0 + (low_pass / cut_off)**(2 * butterworth_order)
        low_pass = 1.0 / denom

    return low_pass


def high_pass(img,cut_off,filter_type='ideal',butterworth_order=1):
    """calculates a highpass filter for an input image

    Args:
        img(np.ndarray): image to calculate filter for
        cut_off (float): cutoff frequency for this filter. units in #TODO
        filter_type (str): the type of filter to apply, 'ideal','gaussian',
            'butterworth'
        butterworth_order(float): butterworth order if butterworth filter is
            being used

    Returns:
        filter(np.ndarray) 2D filter
    """
    return 1 - low_pass(img, cut_off, filter_type, butterworth_order)

# PSF from auto correlation of (exit?) pupil function
def empirical_psf(pupil):
    # from scipy.signal import correlate2d
    # return correlate2d(pupil, pupil, mode='same')
    pupil_fft = np.fft.fftshift(np.fft.fft2(pupil))
    return np.fft.fftshift(np.fft.ifft2(pupil_fft * pupil_fft[::-1,::-1])) # May not need reveresed indices here

if __name__=="__main__":
    import sys

    mode = "mtf"

    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if not (mode.lower() in ["mtf","separable"]):
        raise ValueError("Must run with mode of 'mtf' or 'separable'")

    Nx = Ny = 1024
    super_sampling_factor = 9

    # based on 256x256 input, x3 in each direction for 3x3 sumpsamples per pixel (should be turned into a variable supersampling function)
    arr = np.ones((Ny*super_sampling_factor, Nx*super_sampling_factor)) * 1e-17
    print("Supersampled scene shape:", arr.shape)

    # Instead of a random sampling of deltas, you can instead project an actual radiance map to the focal plane (converted to irradiance) and work with that as input
    # I just need to put the necessary hooks in to make this stuff easy to extend

    deltas = np.random.choice(np.array([False,True]), arr.shape, p=[0.99995, 0.00005])
    n_deltas = deltas.sum()
    print(f"% deltas = {n_deltas / arr.size}")

    irrads = np.random.uniform(1e-15, 1e-17, n_deltas)
    arr[deltas] = irrads
    print(f"Aperture Referred In-Band Irradiance: Mean={arr.mean()}, Max={arr.max()}, Min={arr.min()}")

    # plt.hist(arr.flat, 100, log=True)
    # plt.show()

    # Quick separable convolution test for symmetric kernels
    def separable_conv(im, hfilt, vfilt):
        hres = convolve1d(im, hfilt, mode="constant", axis=1, cval=0)
        return convolve1d(hres, vfilt, mode="constant", axis=0, cval=0)
    
    if mode == "separable":

        kernel = np.array([0.05, 0.05, 0.15, 0.5, 0.15 , 0.05, 0.05])
        filtered = separable_conv(arr, kernel, kernel)

    elif mode == "mtf":

        # test of psf == autocorrelation of aperture, mtf == FFT(psf) -> Currently working!!!
        psf = empirical_psf(low_pass(arr, 5, filter_type='ideal'))
        mtf = np.fft.fftshift(np.fft.fft2(psf / psf.sum())) # We want to use the FFT of the volume-normalized psf array (sums to 1)

        filtered = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(arr)) * mtf))).astype(float)

        # fig, axes = plt.subplots(2,2)
        # axes[0,0].imshow(np.abs(psf))
        # axes[0,1].imshow(np.abs(mtf))
        # axes[1,0].imshow(np.abs(mtf.real))
        # axes[1,1].imshow(np.abs(mtf.imag))
        # plt.show()

    else:
        raise ValueError(f"No known 'mode' matches the one specified ({mode})... exiting now.")

    # Aggregate into "original" 256 x 256 image by summing every chunk of NxN subarrays
    agg = np.sum([np.vsplit(v, filtered.shape[0]//super_sampling_factor) for v in np.hsplit(filtered, filtered.shape[0]//super_sampling_factor)], axis=(2,3)).T

    # Run full pipeline against "actual scene"
    from simple_sensor_model import *

    start = time.time()

    electrons = irrad2electrons(agg)
    print(f"Electrons: Mean={electrons.mean()}, Max={electrons.max()}, Min={electrons.min()}")

    electrons = apply_photo_response_non_uniformity(electrons)
    print(f"Post-PRNU: Mean={electrons.mean()}, Max={electrons.max()}, Min={electrons.min()}")

    electrons = apply_dark_fixed_pattern_noise(electrons, dark_signal(electrons, 10)) # electrons per second, and poisson process applied
    print(f"Post-Dark: Mean={electrons.mean()}, Max={electrons.max()}, Min={electrons.min()}")

    voltage = electrons2voltage(electrons)
    print(f"Voltage: Mean={voltage.mean()}, Max={voltage.max()}, Min={voltage.min()}")
    del electrons

    counts = voltage2counts(voltage, bits=16)
    del voltage

    if isinstance(counts, cp.ndarray):
        counts = counts.get()

    end = time.time()

    print(f"The staring detector array model results in an image of counts given an array of input irradiances.")
    print(f"Total execution time = {end - start} [s]")
    print(f"Estimated frames per second = {1/(end - start)} [fps]")
    print(f"Counts: Mean={counts.mean()}, Max={counts.max()}, Min={counts.min()}")

    # Plot input and result
    fig, axes = plt.subplots(1,3)
    axes[0].imshow(filtered, cmap="inferno")
    axes[1].imshow(counts, cmap="inferno") #, vmin=1000, vmax=counts.mean()+3*counts.std()) #, vmin=counts.min(), vmax=counts.max())
    axes[2].hist(counts.flat, 100, log=True)
    plt.show()


    ###################################################################################
    # Test filtering
    # Test 1: PRF Stack of Gaussians, numpy convolve

    # Test 2: PRF Stack of Gaussians, scipy convolve (FFT)

    # Test 3: Arbitrary FFT + MTF Approach

    # Test 4: With motion in field

    # Test 5: With variable PSF over wide field FOV






    #####################
    # NOTES
    #####################

    # Need to make an equivalent filter in separable and MTF cases, and assert that the total energy on FPA is about equal within some tolerance