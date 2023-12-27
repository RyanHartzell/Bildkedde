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
        self.rsr = vals / vals.max()
        self.center_wl = wl[len(wl)//2]
        self.average_rsr = self.integrate() / (self.wl[-1] - self.wl[0])

        # Optional bspline fit of data to interpolate across for integration. I don't love this, but might have some applications
        # self.spline = 

    def integrate(self, start_index=0, end_index=None):
        return np.trapz(self.rsr[start_index:end_index], x=self.wl)

# Are there any standard modules for doing this?
from sklearn.neighbors import BallTree
class BRDF:
    def __init__(self, theta_i, phi_i, theta_o, phi_o, reflectance, fname=None):
        self.reflectance = None

        # maintain unit sphere of uniform mesh points for NN evaluation

    def evaluate(self, theta, phi):
        pass

    def plot(self):
        pass

class BTDF:
    def __init__(self, theta_i, phi_i, theta_o, phi_o, transmission, fname=None):
        self.transmitance = None

        # maintain unit sphere of uniform mesh points for NN evaluation

    def evaluate(self, theta, phi):
        pass

    def plot(self):
        pass

class BSDF:
    def __init__(self, brdf, btdf, fname=None):
        # maintain unit sphere of uniform mesh points for NN evaluation
        self.brdf = brdf
        self.btdf = btdf

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
        self.norm()

    def __repr__(self):
        return f"Frame({self.x=}, {self.y=}, {self.z=}, {self.origin=})"
    
    def norm(self):
        self.x = self.x / np.linalg.norm(self.x)
        self.y = self.y / np.linalg.norm(self.y)
        self.z = self.z / np.linalg.norm(self.z)

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

        # Normalize direction and Frame axes
        self.frame.norm()
        self.direction = self.direction / np.linalg.norm(self.direction)

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
    
    def get_inverse_camera_matrix(self):
        return np.linalg.inv(self.get_camera_matrix())

    def to(self, point):
        old = self.direction
        self.direction = point - self.position
        self.direction = self.direction / np.linalg.norm(self.direction)

        # Need to update camera frame (should use quaternions for all this shit, but for now get R)

        # From rodigues formula (find axis of rotation via cross, find angle of rotation via dot, apply rotation)
        v = skew_symmetric_cross_product(np.cross(old / np.linalg.norm(old), self.direction))
        R = np.eye(3) + v + np.dot(v,v) * 1 / (1 + np.dot(old / np.linalg.norm(old), self.direction))
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
        fl = focal_length * model_scale # Focal length should be a member!!!!!! That's an important piece of our camera!!!!!

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

    def draw_image_view(self, targets):
        return

############################################################################################################
# Some actual radiometry functions!
############################################################################################################
TOTAL_SOLAR_IRRADIANCE_AT_EARTH = 1.356e5 # W / m^ (RH: Top of atmosphere? Should be at (EARTH_RADIUS-SUN_CENTER)
SUN_RADIUS = 6.957e8 # m
SUN_AREA = np.pi * SUN_RADIUS**2 # m^2
SUN_EARTH_DISTANCE = 1.495978707e11 # 1 AU = 1.495978707e11 m
SOLID_ANGLE_OF_SUN_AT_EARTH = SUN_AREA / SUN_EARTH_DISTANCE
AVG_SOLAR_RADIANCE = TOTAL_SOLAR_IRRADIANCE_AT_EARTH / (SOLID_ANGLE_OF_SUN_AT_EARTH)

# solid angle can be approximated as cross-sectional area / distance for large distances and small effective areas.

# Incoming flux at target
def target_flux(L_sun, A_sun, A_target, distance_sun_to_target):
    return L_sun * A_sun * A_target / distance_sun_to_target**2

# Outgoing radiance from target
def radiance_at_target(target_flux, reflectance, A_target, lambertian=True):
    return target_flux * reflectance / (A_target * (lambertian + 1) * np.pi) # branchless

# Incoming flux at aperture
def aperture_flux(L_target, A_target, sun_angle, A_aperture, distance_target_to_sensor):
    return L_target * A_target * np.cos(sun_angle) * A_aperture / distance_target_to_sensor**2

# I should probably write variants above which take as input just the integrated solid angle of an arbitrary surface wrt some point 
# That being said, this is all you need for an unresolved point source!!!!
# For a resolved source, you really need to sample a bunch of "points" which make up the overall target radiance

# Slight discrepancy with target_flux -> radiance_at_target -> aperture_flux
def irradiance_at_aperture(L_sun, A_sun, A_target, reflectance, solar_angle, distance_sun_to_target, distance_target_to_sensor, lambertian=True):
    return L_sun * A_sun * A_target * np.cos(solar_angle) * reflectance / (distance_sun_to_target**2 * distance_target_to_sensor**2 * (lambertian + 1) * np.pi)

# Helpers
def aperture_area(aperture_radius):
    return np.pi * aperture_radius**2

def approximate_solid_angle(projected_area_at_distance, distance):
    return projected_area_at_distance / distance**2

def effective_focal_length_to_afov(f, detector_array_physical_size):
    return 2 * np.arctan(detector_array_physical_size / (2 * f))

def afov_to_effective_focal_length(afov, detector_array_physical_size):
    return detector_array_physical_size / (2 * np.tan(afov / 2))

# not necessary for small AFOV sensors, but techincally necessary for large format large AFOV sensors.
# constant pixel size in physical units, converted to spherical angles
def effective_focal_length_to_ifov():
    return None

# target to pixel (from CMU notes, but check against Dr. Ientiluci's class notes)
#   if not given effective focal length, need to know pixel IFOV I think???
def irradiance_at_pixel(scene_radiance, focal_length, angle_off_boresight, aperture_area):
    return scene_radiance * aperture_area * np.cos(angle_off_boresight)**4 / (focal_length**2 * 4)

#############################################################################################################
# Test for extremes
# This could also just be (distance_sun_to_target, distance_target_to_sensor, solar_phase_angle, area_cross_section, pixel_ifov, pixel_angle_off_boresight)
def test_irradiance_at_pixel_from_sun_to_target_to_sensor(sun_pos_eci, target_pos_eci, sensor_pos_eci, target_diameter, aperture_area):

    # phi_at_target = target_flux(AVG_SOLAR_RADIANCE, SUN_AREA, )

    # L = 

    # E = irradiance_at_pixel(L, f, db, aperture_area)
    
    # return E

    return None

#############################################################################################################
"""
In this file, we should focus on space targets.
Load object as mesh with associated materials (BSDF)
Illuminate with parallel sun rays (appropriate in space), evaluate photon mapping to get new "sources" (this is global illumination with ambient occlusion, and can be precomputed)
"Illuminate" with inverse sensor rays, evaluate visibility mapping
Evaluate traversal between both sets of probability distribution functions
Radiance constant along rays, but scaled by attenuation/scattering/reflectance at each surface interaction
Average all recieved rays per pixel, casting a random assortment.
Return array of double-precision floating point values for pixel irradiances.

At this point, we have the input to the sensor model!!! This will apply MTF effects and send through sensor effects

"""

# Use this to illuminate target from 
def generate_forward_parallel_sun_ray_bundle(N, unit_vec, radius):
    X = np.random.uniform(-radius,radius,N)
    Y = np.random.uniform(-radius,radius,N)
    return unit_vec + np.c_[X.flat, Y.flat, np.zeros(N)]

# The sensor ray bundle should be a set of rays exiting a pixel into that pixel's IFOV footprint in the scene
# For now we use uniform random sampling by default, but Sobol or Hamming patterns may be more numerically stable from a sampling perspective
# This hinges on pinhole camera basics
def generate_backward_pixel_ray_bundle(camera, pixel_x, pixel_y, method="uniform"):
    return

def generate_importance_bsdf_rays():
    return

if __name__=="__main__":
    c = Camera(np.array([0,0,0]))
    # target = np.array([20,0,0])
    target = np.array([20,10,-3])
    c.to(target)
    print(c.direction)
    print(c.calc_aer(target))

    import matplotlib.pyplot as plt

    # fig = plt.figure()
    # ax = fig.gca(projection="3d")

    N = 100
    targets = np.random.uniform(-100,100, size=(100,3))

    from matplotlib.animation import FuncAnimation

    # def update(i):
    #     t = targets[i]
    #     c.to(t)

    #     ax.cla()
    #     ax.scatter(t[0], t[1], t[2])
    #     c.draw_camera_model(ax, model_scale=5)

    #     ax.set_xlim(-110,110)
    #     ax.set_ylim(-110,110)
    #     ax.set_zlim(-110,110)

    # ani = FuncAnimation(fig, update, frames=range(N))
    # plt.show()

    # Seems like my sensor pointing thing is working!!!!

    # Grab random AzEl? Or random angular sep from boresight (direction vector)?
    FOV = 5 * np.pi / 180.

    # THis is actually definitely valid, but for actual boresight you need to rotate back to az el and your camera orientation might be funky (not perfectly aligned along az and el...)
    # az = np.random.uniform(-0.5, 0.5, size=(N)) * FOV + 90.0 # Boresight @ az=90, el=0.0
    # el = np.random.uniform(-0.5, 0.5, size=(N)) * FOV + 90.0
    # ra = np.random.uniform(20,50, size=(N))

    # def aer2world(az, el, ra=1.0):
    #     x = ra * np.sin(az) * np.cos(el)
    #     y = ra * np.sin(az) * np.sin(el)
    #     z = ra * np.cos(az)

    #     return np.array([x,y,z])

    # Might have some cos and sin flippage here...
    def aer2world(az, el, ra=1.0):
        x = ra * np.sin(el) * np.cos(az)
        y = ra * np.sin(el) * np.sin(az)
        z = ra * np.cos(el)
        v = np.array([x,y,z])
        return v / np.linalg.norm(v, axis=0)

    # Let's try it in boresight + (unit distance) range (LOS)...
    # This is really UV coordinate conversion
    bx = np.random.uniform(-0.5, 0.5, size=(N)) * FOV # Boresight @ az=90, el=0.0
    by = np.random.uniform(-0.5, 0.5, size=(N)) * FOV
    # r = np.random.uniform(10, 40, size=N)
    # r = np.ones_like(bx)

    # Inverse of below, likely rotate camera to azel from camera coords
    # THESE ARE ANGLES!!! -> becomes unit vector
    def boresight2camera(bx, by):
        v = np.array([np.tan(bx), np.tan(by), np.ones_like(bx)]) # tan(theta) = Opp / Adj = Opp / 1.0
        return v / np.linalg.norm(v, axis=0)

    # Project onto camera FPA with proper components (X,Y if that's our plane, with Z out along LOS)
    # Get difference in camera X and camera Y from point to boresight (Can also do this directly in AZEL but need spherical coordinate great circle distance)
    def camera2boresight(target):
        bx = np.arctan2(target[0], target[2]) # atan2(x / z)
        by = np.arctan2(target[1], target[2]) # atan2(x / z)
        return bx, by
    
    # Project camera frame to image plane with SIMPLE IDEAL PERSPECTIVE TRANSFORM
    def camera2image_plane(target, efl=1.0):
        x = efl * target[0] / target[2]
        y = efl * target[1] / target[2]
        return x,y
    
    # We want one which takes actual world coords, transforms to camera frame, and then applies boresight2cam_frame_2boresight
    def world2boresight():
        # cam_pts = cam_mat @ world_pts        
        # return camera2boresight(cam_pts)
        
        return

    # Full world -> image space perspective transform
    def world2image_plane():
        # cam_pts = cam_mat @ world_pts        
        # return camera2image_plane(cam_pts)

        return

    # Convert boresight coordinates to world coordinates by using angle from direction vector (boresight) and provided range along that vector
    cam_pts = np.array(boresight2camera(bx, by))
    print(cam_pts.shape)

    # Transform camera points back to world points
    world_pts = c.get_inverse_camera_matrix().T @ np.vstack([cam_pts, np.ones_like(cam_pts[0])]) # Make homogenous coordinates [x, y, z, 1]
    print(world_pts.shape)

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    ax.scatter(4*world_pts[0], 4*world_pts[1], 4*world_pts[2], c="orange", alpha=0.2)
    ax.scatter(20*world_pts[0], 20*world_pts[1], 20*world_pts[2])
    ax.plot([0.0, c.direction[0]*20], [0.0, c.direction[1]*20], [0.0, c.direction[2]*20], '--r')
    c.draw_camera_model(ax, 2, 2)
    ax.set_xlim(-50,50)
    ax.set_ylim(-50,50)
    ax.set_zlim(-50,50)
    plt.show()

    # I GOT IT WORKING!!!!!!!!!!!!!!!!!!!!!!!!!
    new_cam_pts = c.get_camera_matrix().T @ np.vstack([targets.T, np.ones(len(targets))])
    print(new_cam_pts.shape)
    new_image_pts = camera2image_plane(new_cam_pts[:,np.where(new_cam_pts[2] > 4.0)], 4.0) # EFL of 4 units
    # afov = effective_focal_length_to_afov(4.0, 1.)
    # print(afov)

    from matplotlib.patches import Rectangle
    fig, axs = plt.subplots(1,2)
    axs[0].scatter(new_image_pts[0], new_image_pts[1])
    mask = (new_image_pts[0] > -2) & (new_image_pts[0] < 2) & (new_image_pts[1] > -2) & (new_image_pts[1] < 2)
    print(mask.shape)
    axs[0].scatter(new_image_pts[0][mask], new_image_pts[1][mask], marker='x', c='r')
    rect = Rectangle((-2,-2), 4., 4., fill='grey', alpha=0.2) # Show bounds of our focal plane array
    axs[0].add_artist(rect)
    axs[1].axis('off')

    ax3d = fig.add_subplot(122, projection='3d')
    
    targets = targets[np.where(new_cam_pts[2] > 4.0)]
    print(targets.shape)

    ax3d.scatter3D(*targets[~np.squeeze(mask)].T[:])
    ax3d.scatter3D(*targets[np.squeeze(mask)].T[:], c='r')
    c.draw_camera_model(ax3d)
    plt.show()

    # Now show a 3D plot of world and camera with a boresight/image plane plot to the right
    
    # targets = 

    # def update(i):
    #     t = targets[i]
    #     c.to(t)

    #     ax.cla()
    #     ax.scatter(t[0], t[1], t[2])
    #     c.draw_camera_model(ax, model_scale=5)

    #     ax.set_xlim(-110,110)
    #     ax.set_ylim(-110,110)
    #     ax.set_zlim(-110,110)

    # ani = FuncAnimation(fig, update, frames=range(N))
    # plt.show()

    ####################################################################################
    # Hook into point source and sensor model?

    from point_source_test import *
    from simple_sensor_model import *

    # Create a bunch of world frame XYZ points with associated  
    bx = np.random.uniform(-np.pi/8, np.pi/8, 1000)
    by = np.random.uniform(-np.pi/8, np.pi/8, 1000)
    r = np.random.uniform(5000, 50000, 1000) # units of km

    # Setup a camera
    cam = Camera([0,0,0])

    # Project boresight pts to camera frame
    # Technically I can go straight from boresight to image...
    cam_pts = np.array(boresight2camera(bx, by)) * r # scale out to actual distance along unit vector direction

    # Project to image plane
    img_pts = camera2image_plane(cam_pts, efl=1.0)

    # Show img_pts in image plane
    plt.scatter(img_pts[0], img_pts[1], c=r)
    plt.show()

    # Now run these points through sensor model given sun angle and range and target size and albedo...
    # irrads = 

    ####################################################################################
    # Do the same using a celestrak catalog
    # XYZ in time -> Set up camera on some arbitrary host satellite -> Subsample in XYZ or boresight -> Calculate irradiance @ aperture -> Convert to Image Space -> Send through sensor model
