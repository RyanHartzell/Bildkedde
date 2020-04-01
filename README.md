# Bildkedde
An extensible imaging system simulation library utilizing an MTF and image chain approach.

# Roadmap

1) Basic algorithms and classes with routines in numpy/scipy for spatial/time<->frequency domain transforms on the fly 
2) Dedicated Image or Array object for describing slices of any step of the optical imaging chain at any point during processing
3) Actual sensor model objects (both high-level for beginners + API simplicity, and low-level for flexibility - including some factories and 'class converter' decorators!!!)
4) Providing some sort of linking structure (networkx or possibly utilizing IP)
5) Improvements to the API + simplifying common routines to single callables
***6) LISTED LAST, BUT MOST IMPORTANT: Integrate sensor model standards from other fields and work on white paper + documentation describing and citing process IN DETAIL 
 
# TODO

- Fourier transform stuff for far-field (Fraunhofer) imaging
- Can I even bother with (Fresnel) near field for this?
- Investigate available python optics libraries
- Image/Response object
- Object/Impulse object
- MTF/OTF object
- PSF object
- PRF object
- HDF5 (or numpy) LUT support
- Element object (basically a processor instance. __call__ func stores the primary input as the impulse, and processing can also be delayed or run immediately (by default). Has start/impulse, end/response, transfer properties, all of which are objects with spatial and frequency properties, and also a during/z/t(normalized_z_or_time_slice) which can be implemented to render the effects of transfer function as the wavefront/impulse travels through space/time/the element.
- System object (consists of our graph of optical/electronic/modifier/mathematical elements) -> This will be base for Optics/FPA/Readout
