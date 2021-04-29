# Bildkedde
An extensible imaging system simulation library utilizing an MTF and image chain approach.

# Roadmap

1) Basic algorithms and classes with routines in numpy/scipy for spatial/time<->frequency domain transforms on the fly
2) Dedicated Image or Array object for describing slices of any step of the optical imaging chain at any point during processing
3) Actual sensor model objects (both high-level for beginners + API simplicity, and low-level for flexibility - including some factories and 'class converter' decorators!!!)
4) Providing some sort of linking structure (networkx or possibly utilizing IP)
5) Improvements to the API + simplifying common routines to single callables
6) __LISTED LAST, BUT MOST IMPORTANT:__ Integrate sensor model standards from other fields and work on white paper + documentation describing and citing process IN DETAIL

# TODO

- Fourier transform stuff for far-field (Fraunhofer) imaging
- Can I even bother with (Fresnel) near field for this?
- Investigate available python optics libraries (for now we're assuming abberations and distortion are calculated independently)
- Image object
- Object object
- MTF(abs)/OTF(complex) object
- PSF object
- PRF/PixelTF object (requires knowledge of FPA properties such as sampling and pixel dimensions, but useful for analytics and processing tricks!)
- HDF5 (or numpy) LUT support
- Element object (basically a processor instance. __call__ func stores the primary input as the impulse, and processing can also be delayed or run immediately (by default). Has input/impulse/object, output/response/image, transfer properties, all of which are objects with spatial and frequency properties, and also a during/z/t(normalized_z_or_time_slice) which can be implemented to render the effects of space-time-dependent transfer function as the wavefront/input travels through the element.
- System object (consists of our graph of optical/electronic/modifier/mathematical elements) -> This will be base for Optics/FPA/Readout

# Library Structure

* Core
  * Fourier Transforms
  * Fraunhofer and Fresnel support
  * Randomness/Noise
  * Basic array math support
  * GPU array support (CUDA Python cupy, pytorch, and/or Numba support?)
  * Base interfaces to inherit from
    * Chain object
    * Link object
* Image
* Noise
* Optics
* Radiometry
* Sensor
