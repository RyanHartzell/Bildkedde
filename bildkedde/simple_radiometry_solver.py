import numpy as np

def check_orthagonal(a, b):
    return np.isclose(np.dot(a,b), 0.0, atol=1e-20)

def skew_symmetric_cross_product(v):
    """this function returns a numpy array with the skew symmetric cross product matrix for vector.
    the skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)
    
    Ref: https://stackoverflow.com/questions/36915774/form-numpy-array-from-possible-numpy-array"""

    return np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])

# Is this provided in pyspectral?
class Spectrum:
    def __init__(self, wl, vals) -> None:
        self.wl = wl
        self.vals = vals
        self.center_wl = wl[len(wl)//2]
        self.average_val = self.integrate() / len(self.wl)

        # Optional bspline fit of data to interpolate across for integration. I don't love this, but might have some applications
        # self.spline = 

    def integrate(self, start_index=0, end_index=-1):
        return np.trapz(self.vals[start_index:end_index])

# Are there any standard modules for doing this?
class BSDF:
    def __init__(self, reflectance, transmitance, fname=None):
        self.reflectance = None
        self.transmitance = None

        # maintain unit sphere of uniform mesh points for NN evaluation

        # 

    def evaluate(self, theta, phi):
        pass

    def plot(self):
        pass

class SurfaceInterface:
    ni = 1.0
    no = 1.0

# Certain I don't need to make this homegrown, but a good exercise!
class Frame:
    def __init__(self, x, y, z=None, origin=np.array([0,0,0])) -> None:
        # Right-handed system by default
        if not check_orthagonal(x,y):
            raise ValueError("Frame axes must be orthagonal in R3")

        self.x = np.asarray(x)
        self.y = np.asarray(y)

        self.z = np.asarray(z) if z is not None else np.cross(x, y)
        self.origin = origin

    def rotate(self, R):        
        # Assemble and transpose since R operates on column vectors such that: Nx3 @ 3x3 -> 
        d = np.asarray([self.x, self.y, self.z]) @ R
        self.x = d[0]
        self.y = d[1]
        self.z = d[2]

    def __repr__(self):
        return f"Frame({self.x=}, {self.y=}, {self.z=}, {self.origin=})"

XHAT = np.array([1,0,0])
YHAT = np.array([0,1,0])
ZHAT = np.array([0,0,1])
WORLD = Frame(XHAT, YHAT, ZHAT)

class Camera:
    def __init__(self, position, direction=np.array([1,0,0]), camera_normal=np.array([0,0,1]), camera_frame=None, aer_frame=Frame([1,0,0], [0,1,0], [0,0,1])) -> None:
        self.position = position

        if isinstance(camera_frame, Frame):
            self.direction = camera_frame.z # for camera frame, this is +z axis. +x and +y can then be defined via the focal plane or image plane
            self.normal = camera_frame.y # might allow specifying the local camera normal in order to define camera frame or allow for rotating AER frame via that
            self.frame = camera_frame

        else:
            self.direction = direction # for camera frame, this is +z axis. +x and +y can then be defined via the focal plane or image plane
            self.normal = camera_normal
            self.frame = Frame(np.cross(camera_normal, direction), camera_normal, direction)

        # Should probably just specify which frame to use during AER calculation...
        self.aer_frame = aer_frame # WCS? or platform? or camera home(init frame)? I like time dependent platform Frame personally...

        # Set up camera matrix
        self.camera_matrix = self.get_camera_matrix()

    def get_camera_matrix(self):
        # Compute camera matrix at current R (rotation from world frame) and t (translation from world origin)
        world = WORLD

        # From rodigues formula (find axis of rotation via cross, find angle of rotation via dot, apply rotation)
        v = skew_symmetric_cross_product(np.cross(world.z / np.linalg.norm(world.z), self.direction / np.linalg.norm(self.direction)))
        R = np.eye(3) + v + np.dot(v,v) * 1 / (1 + np.dot(world.z / np.linalg.norm(world.z), self.direction / np.linalg.norm(self.direction)))

        # C = -R.T @ self.position # This might actually be for points in world? although not sure...

        C = np.eye(4)
        C[:3, :3] = R
        C[:3, 3] = self.position
        return C

    def to(self, point):
        old = self.direction
        self.direction = point - self.position

        # Need to update camera frame (should use quaternions for all this shit, but for now get R)

        # From rodigues formula (find axis of rotation via cross, find angle of rotation via dot, apply rotation)
        v = skew_symmetric_cross_product(np.cross(old / np.linalg.norm(old), self.direction / np.linalg.norm(self.direction)))
        R = np.eye(3) + v + np.dot(v,v) * 1 / (1 + np.dot(old / np.linalg.norm(old), self.direction / np.linalg.norm(self.direction)))
        self.frame.rotate(R) # Operates inplace
        self.camera_matrix = self.get_camera_matrix()

    def translate(self, delta):
        self.position += delta
        self.frame.origin = self.position 

    # This function should really be for rotating the camera's frame itself....
    def rotate(self, R):
        # Euler transform. Should really change this to emulate gimbal system (2axis=no rotation around boresight, or 3axis)
        self.frame = self.frame.rotate(R)
        self.direction = self.frame.z

    def calc_aer(self, target, frame=None):
        # Defaults to camera frame if frame is None
        if frame is None:
            frame = self.frame

        los = target - self.position
        range = np.linalg.norm(los)
        
        el = np.arccos(np.dot(los, frame.z) / (range*np.linalg.norm(frame.z))) * 180./np.pi
        az = np.arctan2(np.dot(los, frame.y) / (range*np.linalg.norm(frame.y)), np.dot(los, frame.x) / (range*np.linalg.norm(frame.x))) * 180./np.pi

        return az, el, range
        
    # CV2/samples/python/camera_calibration_show_extrinsics.py
    def draw_camera_model(self, ax, fovx=1, fovy=1, focal_length=4, model_scale=1):
        cam = self.camera_matrix
        print(cam)
        print(cam.shape)

        w = fovx / 2 * model_scale
        h = fovy / 2 * model_scale
        fl = focal_length * model_scale

        # build it in a nice normal space with +z as boresight, position at world, then transform with R|t
        model = np.array([WORLD.origin,
                      WORLD.origin + [w,h,fl],
                      WORLD.origin + [-w,h,fl],
                      WORLD.origin,
                      WORLD.origin + [-w,h,fl],
                      WORLD.origin + [-w,-h,fl],
                      WORLD.origin,
                      WORLD.origin + [-w,-h,fl],
                      WORLD.origin + [w,-h,fl],
                      WORLD.origin,
                      WORLD.origin + [w,-h,fl],
                      WORLD.origin + [w,h,fl],
                      WORLD.origin])
        
        model = np.hstack([model, np.ones((len(model), 1))])
        sensor = cam @ model.T

        ax.plot(sensor[0,:], sensor[1,:], sensor[2,:], 'r')

############################################################################################################
# Some actual radiometry functions!
############################################################################################################
TOTAL_SOLAR_IRRADIANCE_AT_EARTH = 1.356e5 # W / m^2
SUN_RADIUS = 6.957e8 # m
SUN_AREA = np.pi * SUN_RADIUS**2 # m^2
SUN_EARTH_DISTANCE = 1.495978707e11 # 1 AU = 1.495978707e11 m
SOLID_ANGLE_OF_SUN_AT_EARTH = SUN_AREA / SUN_EARTH_DISTANCE
TOTAL_SOLAR_RADIANCE_AT_EARTH = TOTAL_SOLAR_IRRADIANCE_AT_EARTH / (SOLID_ANGLE_OF_SUN_AT_EARTH)

# solid angle can be approximated as cross-sectional area / distance for large distances and small effective areas.

def target_flux(L_sun, A_sun, A_target, distance_sun_to_target):
    return L_sun * A_sun * A_target / distance_sun_to_target**2

def radiance_at_target(target_flux, reflectance, A_target, lambertian=True):
    return target_flux * reflectance / (A_target * (lambertian + 1) * np.pi) # branchless

def aperture_flux(L_target, A_target, sun_angle, A_aperture, distance_target_to_sensor):
    return L_target * A_target * np.cos(sun_angle) * A_aperture / distance_target_to_sensor**2

# I should probably write variants above which take as input just the integrated solid angle of an arbitrary surface wrt some point 
# That being said, this is all you need for an unresolved point source!!!!
# For a resolved source, you really need to sample a bunch of "points" which make up the overall target radiance
def irradiance_at_aperture(L_sun, A_sun, A_target, reflectance, solar_angle, distance_sun_to_target, distance_target_to_sensor, lambertian=True):
    return L_sun * A_sun * A_target * np.cos(solar_angle) * reflectance / (distance_sun_to_target**2 * distance_target_to_sensor**2 * (lambertian + 1) * np.pi)

# Helper
def aperture_area(aperture_diameter):
    return np.pi * aperture_diameter**2

# Assumes radial symmetry, should extend to rectangular arrays with different effective afov
def effective_focal_length_to_afov(f, detector_array_physical_size):
    return 2 * np.arctan(detector_array_physical_size / (2 * f))

def afov_to_effective_focal_length(afov, detector_array_physical_size):
    return detector_array_physical_size / (2 * np.tan(afov / 2))

# target to pixel (from CMU notes, but check against Dr. Ientiluci's class notes)
#   if not given effective focal length, need to know pixel IFOV I think???
def irradiance_at_pixel(scene_radiance, focal_length, angle_off_boresight, aperture_area):
    return scene_radiance * aperture_area * np.cos(angle_off_boresight)**4 / (focal_length**2 * 4)

#############################################################################################################
# Test for extremes
# This could also just be (distance_sun_to_target, distance_target_to_sensor, solar_phase_angle, area_cross_section, pixel_ifov, pixel_angle_off_boresight)
def test_irradiance_at_pixel_from_sun_to_target_to_sensor(sun_pos_ecef, target_pos_ecef, sensor_pos_ecef, target_diameter, aperture_area):
    return None

if __name__=="__main__":
    c = Camera(np.array([0,0,0]))
    target = np.array([20,0,0]) # +20x
    c.to(target)
    print(c.calc_aer(target))

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    N = 100
    targets = np.random.uniform(-100,100, size=(100,3))

    from matplotlib.animation import FuncAnimation

    def update(i):
        t = targets[i]
        c.to(t)

        ax.cla()
        ax.scatter(t[0], t[1], t[2])
        c.draw_camera_model(ax, model_scale=5)

        ax.set_xlim(-110,110)
        ax.set_ylim(-110,110)
        ax.set_zlim(-110,110)

    ani = FuncAnimation(fig, update, frames=range(N))
    plt.show()

    # Seems like my sensor pointing thing is working!!!!

    # Grab random AzEl? Or random angular sep from boresight (direction vector)?
    FOV = 5 * np.pi / 180.

    # THis is actually definitely valid, but for actual boresight you need to rotate back to az el and your camera orientation might be funky (not perfectly aligned along az and el...)
    # az = np.random.uniform(-0.5, 0.5, size=(N)) * FOV + 90.0 # Boresight @ az=90, el=0.0
    # el = np.random.uniform(-0.5, 0.5, size=(N)) * FOV + 90.0
    # ra = np.random.uniform(20,50, size=(N))

    # x = ra * np.sin(az) * np.cos(el)
    # y = ra * np.sin(az) * np.sin(el)
    # z = ra * np.cos(az)

    # Let's try it in boresight + (unit distance) range (LOS)...
    # This is really UV coordinate conversion
    bx = np.random.uniform(-0.5, 0.5, size=(N)) * FOV # Boresight @ az=90, el=0.0
    by = np.random.uniform(-0.5, 0.5, size=(N)) * FOV
    # r = np.random.uniform(10, 40, size=N)
    r = np.ones_like(bx)

    # Inverse of below, likely rotate camera to azel from camera coords and then convert to XYZ via above with bogus distance of 1
    def boresight2world():
        return

    # Easy one:
    # Convert coordinate from XYZ to AZEL
    # Project onto camera FPA with proper components (X,Y if that's our plane, with Z out along LOS)
    # Get difference in camera X and camera Y from point to boresight (Can also do this directly in AZEL but need spherical coordinate great circle distance)
    def world2boresight():
        return

    # Convert boresight coordinates to world coordinates by using angle from direction vector (boresight) and provided range along that vector
    x, y, z = boresight2world(bx, by, r)

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    ax.scatter(x, y, z)
    c.draw_camera_model(ax, 2, 2)
    ax.set_xlim(-50,50)
    ax.set_ylim(-50,50)
    ax.set_zlim(-50,50)
    plt.show()