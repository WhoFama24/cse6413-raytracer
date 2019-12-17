import random
import string
from math import inf
import numpy as np


class Volume:
    """
    class Volume:

        Stores the six planes that define a volumetric space.
        Each plane is easily accessible from its Python decorator.

    All parameters must be scalar (int/float) values and default to the clip-space volume if no argument is provided on
    construction.
    """

    def __init__(self, left=-1, right=1, bottom=-1, top=1, near=0.1, far=inf):
        if not all(isinstance(in_var, (int, float)) for in_var in [left, right, bottom, top, near, far]):
            raise TypeError("Expected int or float.")

        self.__left = left
        self.__right = right
        self.__bottom = bottom
        self.__top = top
        self.__near = near
        self.__far = far

    @property
    def left(self):
        return self.__left

    @left.setter
    def left(self, val):
        if not isinstance(val, (int, float)):
            raise TypeError("Expected int or float.")
        self.__left = val

    @property
    def right(self):
        return self.__right

    @right.setter
    def right(self, val):
        if not isinstance(val, (int, float)):
            raise TypeError("Expected int or float.")
        self.__right = val

    @property
    def bottom(self):
        return self.__bottom

    @bottom.setter
    def bottom(self, val):
        if not isinstance(val, (int, float)):
            raise TypeError("Expected int or float.")
        self.__bottom = val

    @property
    def top(self):
        return self.__top

    @top.setter
    def top(self, val):
        if not isinstance(val, (int, float)):
            raise TypeError("Expected int or float.")
        self.__top = val

    @property
    def near(self):
        return self.__near

    @near.setter
    def near(self, val):
        if not isinstance(val, (int, float)):
            raise TypeError("Expected int or float.")
        self.__near = val

    @property
    def far(self):
        return self.__far

    @far.setter
    def far(self, val):
        if not isinstance(val, (int, float)):
            raise TypeError("Expected int or float.")
        self.__far = val


class Point:
    """
    class Point:

        Represents a standard graphics point with position coordinates (x,y,z), normal coordinates (nx,ny,nz) and
        texture coordinates (u,v). Each element of the point is conveniently exposed with appropriate Python decorators.

    All parameters must be scalar (int/float) values and default to 0 if no argument is provided on construction.

    Point(x, y, z, nx, ny, nz, u, v)
    """

    def __init__(self, pos=np.zeros(3), normals=np.zeros(3), rgba=np.zeros(4),
                 ambient_level=0.05, specular_exponent=100, reflection=np.zeros(4)):
        if isinstance(pos, list):
            if not len(pos) == 3:
                raise ValueError("Position should be a length-3 vector")
            pos = np.array(pos)
        elif isinstance(pos, np.ndarray):
            if not np.size(pos) == 3:
                raise ValueError("Position should be a length-3 vector")
        else:
            raise TypeError("Unexpected type for position")

        if isinstance(normals, list):
            if not len(normals) == 3:
                raise ValueError("Normals should be a length-3 vector")
            normals = np.array(normals)
        elif isinstance(normals, np.ndarray):
            if not np.size(normals) == 3:
                raise ValueError("Normals should be a length-3 vector")
        else:
            raise TypeError("Unexpected type for position")

        if isinstance(rgba, list):
            if not len(rgba) == 4:
                raise ValueError("RGBA should be a length-4 vector")
            rgba = np.array(rgba)
        elif isinstance(rgba, np.ndarray):
            if not np.size(rgba) == 4:
                raise ValueError("RGBA should be a length-4 vector")
        else:
            raise TypeError("Unexpected type for position")

        if isinstance(reflection, list):
            if not len(reflection) == 4:
                raise ValueError("Reflection color should be a length-4 vector")
            reflection = np.array(reflection)
        elif isinstance(reflection, np.ndarray):
            if not np.size(reflection) == 4:
                raise ValueError("Reflection color should be a length-4 vector")
        else:
            raise TypeError("Unexpected type for reflection color")

        self.__point = {'x': pos[0],
                        'y': pos[1],
                        'z': pos[2],
                        'normal_x': normals[0],
                        'normal_y': normals[1],
                        'normal_z': normals[2],
                        'color_r': rgba[0],
                        'color_g': rgba[1],
                        'color_b': rgba[2],
                        'color_a': rgba[3],
                        'reflection_r': reflection[0],
                        'reflection_g': reflection[1],
                        'reflection_b': reflection[2],
                        'reflection_a': reflection[3]
                        }
        self.__ambient_level = ambient_level
        self.__specular_exponent = specular_exponent

    def transform(self, transform: np.ndarray):
        """
        Transforms the point position and normal.

        :param transform:
            (ndarray) 4x4 transformation array

        :return:
            (Point) returns object instance for convenience in calculations
        """
        if not isinstance(transform, np.ndarray):
            raise TypeError("Unexpected type for transform")
        if not transform.shape == (4, 4):
            raise ValueError("Transform should be a 4x4 array")

        point = np.array([self.x, self.y, self.z, 1])
        normal = np.array([self.nx, self.ny, self.nz, 1])
        self.position = np.matmul(transform, point)[0:3]
        self.normal = np.matmul(transform, normal)[0:3]
        return self

    @property
    def x(self) -> float:
        return self.__point['x']

    @x.setter
    def x(self, val):
        if not isinstance(val, (int, float)):
            raise ValueError("Expected an integer or floating-point value")
        self.__point['x'] = float(val)

    @property
    def y(self) -> float:
        return self.__point['y']

    @y.setter
    def y(self, val):
        if not isinstance(val, (int, float)):
            raise ValueError("Expected an integer or floating-point value")
        self.__point['y'] = float(val)

    @property
    def z(self) -> float:
        return self.__point['z']

    @z.setter
    def z(self, val):
        if not isinstance(val, (int, float)):
            raise ValueError("Expected an integer or floating-point value")
        self.__point['z'] = float(val)

    @property
    def nx(self) -> float:
        return self.__point['normal_x']

    @nx.setter
    def nx(self, val):
        if not isinstance(val, (int, float)):
            raise ValueError("Expected an integer or floating-point value")
        self.__point['normal_x'] = float(val)

    @property
    def ny(self) -> float:
        return self.__point['normal_y']

    @ny.setter
    def ny(self, val):
        if not isinstance(val, (int, float)):
            raise ValueError("Expected an integer or floating-point value")
        self.__point['normal_y'] = float(val)

    @property
    def nz(self) -> float:
        return self.__point['normal_z']

    @nz.setter
    def nz(self, val):
        if not isinstance(val, (int, float)):
            raise ValueError("Expected an integer or floating-point value")
        self.__point['normal_z'] = float(val)

    @property
    def r(self) -> float:
        return self.__point['color_r']

    @r.setter
    def r(self, val):
        if not isinstance(val, (int, float)):
            raise ValueError("Expected an integer or floating-point value")
        self.__point['color_r'] = float(val)

    @property
    def g(self) -> float:
        return self.__point['color_g']

    @g.setter
    def g(self, val):
        if not isinstance(val, (int, float)):
            raise ValueError("Expected an integer or floating-point value")
        self.__point['color_g'] = float(val)

    @property
    def b(self) -> float:
        return self.__point['color_b']

    @b.setter
    def b(self, val):
        if not isinstance(val, (int, float)):
            raise ValueError("Expected an integer or floating-point value")
        self.__point['color_b'] = float(val)

    @property
    def a(self) -> float:
        return self.__point['color_a']

    @a.setter
    def a(self, val):
        if not isinstance(val, (int, float)):
            raise ValueError("Expected an integer or floating-point value")
        self.__point['color_a'] = float(val)

    @property
    def reflection_r(self) -> float:
        return self.__point['reflection_r']

    @property
    def reflection_g(self) -> float:
        return self.__point['reflection_g']

    @property
    def reflection_b(self) -> float:
        return self.__point['reflection_b']

    @property
    def reflection_a(self) -> float:
        return self.__point['reflection_a']

    @property
    def position(self) -> np.ndarray:
        return np.array([self.__point['x'], self.__point['y'], self.__point['z']])

    @position.setter
    def position(self, val):
        if isinstance(val, list):
            if not len(val) == 3:
                raise ValueError("Position requires a 3-element array for x,y,z coordinates")
        if isinstance(val, np.ndarray):
            if not np.size(val) == 3:
                raise ValueError("Position requires a 3-element array for x,y,z coordinates")

        self.__point['x'] = float(val[0])
        self.__point['y'] = float(val[1])
        self.__point['z'] = float(val[2])

    @property
    def normal(self) -> np.ndarray:
        return np.array([self.__point['normal_x'], self.__point['normal_y'], self.__point['normal_z']])

    @normal.setter
    def normal(self, val):
        self.__point['normal_x'] = float(val[0])
        self.__point['normal_y'] = float(val[1])
        self.__point['normal_z'] = float(val[2])

    @property
    def rgb(self) -> np.ndarray:
        return np.array([self.__point['color_r'], self.__point['color_g'], self.__point['color_b']])

    @rgb.setter
    def rgb(self, val):
        if not isinstance(val, list):
            if not len(val) == 3:
                raise ValueError("Expected color vector of length 3")
        self.__point['color_r'] = float(val[0])
        self.__point['color_g'] = float(val[1])
        self.__point['color_b'] = float(val[2])

    @property
    def rgba(self) -> np.ndarray:
        return np.array([self.__point['color_r'],
                         self.__point['color_g'],
                         self.__point['color_b'],
                         self.__point['color_a']])

    @rgba.setter
    def rgba(self, val):
        if not isinstance(val, list):
            if not len(val) == 4:
                raise ValueError("Expected color vector of length 4")
        self.__point['color_r'] = float(val[0])
        self.__point['color_g'] = float(val[1])
        self.__point['color_b'] = float(val[2])
        self.__point['color_a'] = float(val[3])

    @property
    def ambient(self) -> np.ndarray:
        return self.__ambient_level * self.diffuse

    @property
    def diffuse(self) -> np.ndarray:
        return self.rgb

    @property
    def specular_exponent(self):
        return self.__specular_exponent

    @property
    def reflection_rgba(self) -> np.ndarray:
        return np.array([
            self.__point['reflection_r'],
            self.__point['reflection_g'],
            self.__point['reflection_b'],
            self.__point['reflection_a']
        ])

    @property
    def reflection_rgb(self) -> np.ndarray:
        return np.array([
            self.__point['reflection_r'],
            self.__point['reflection_g'],
            self.__point['reflection_b']
        ])


class Ray:
    MAX_DEPTH = 3

    def __init__(self, eye=None, direction=None):
        if isinstance(eye, list):
            if not len(eye) == 3:
                raise ValueError("Eye coordinates should be a length-3 list.")
            self.__eye = Point(np.array(eye))
        elif isinstance(eye, np.ndarray):
            if not np.size(eye) == 3:
                raise ValueError("Eye coordinates should be a length-3 vector.")
            self.__eye = Point(eye)
        elif isinstance(eye, Point):
            self.__eye = eye
        elif eye is None:
            self.__eye = Point()
        else:
            raise TypeError("Unexpected input for eye")

        if isinstance(direction, list):
            if not len(direction) == 3:
                raise ValueError("Direction should be a length-3 list or ndarray")
            self.__dir = np.array(direction)
        elif isinstance(direction, np.ndarray):
            if not np.size(direction) == 3:
                raise ValueError("Direction should be a length-3 list or ndarray")
            self.__dir = direction
        else:
            raise TypeError("Unexpected input for target")

    @property
    def e(self) -> Point:
        return self.__eye

    @e.setter
    def e(self, val):
        if isinstance(val, list):
            # Error handling for list length is passed to Point
            self.__eye.position = val
        elif isinstance(val, Point):
            self.__eye = val
        else:
            raise TypeError("Unexpected type for eye")

    @property
    def d(self) -> np.ndarray:
        return self.__dir

    @d.setter
    def d(self, val):
        if isinstance(val, list):
            # Error handling for list length is passed to Point
            self.__dir.position = val
        elif isinstance(val, Point):
            self.__dir = val
        else:
            raise TypeError("Unexpected type for direction")

    def propagate(self, scene, depth=0) -> np.ndarray:
        # Return background color if ray has reached max bounces
        if depth > Ray.MAX_DEPTH:
            return Point().rgb

        # Trace the ray through the scene and find the intersection if it exists
        t, intersection = scene.trace(self)
        if t == inf:
            return Point().rgb

        # # Perform shading calculations
        diffuse_accumulator = np.zeros(3)
        specular_accumulator = np.zeros(3)
        for light in scene.lights:
            # Shadow feeler ray
            # light_direction = light.position - intersection.position
            light_direction = light.position
            light_ray = Ray(eye=intersection.position, direction=-light_direction)
            # t_light = np.linalg.norm((light.position - light_ray.e.position) / light_ray.d)
            t_shadow, intersection_shadow = scene.trace(light_ray)

            # if t_shadow > t_light:
            if t_shadow == inf:
                light_normal = -light_direction / np.linalg.norm(light_direction)
                model_normal = intersection.normal / np.linalg.norm(intersection.normal)
                view_normal = -self.d / np.linalg.norm(self.d)

                if np.dot(light_normal, model_normal) > 0.0:
                    # Diffuse Lighting
                    diffuse_accumulator += intersection.diffuse * light.diffuse * np.dot(light_normal, model_normal)

                    # Specular Lighting
                    halfway_vector = (light_normal + view_normal) / np.linalg.norm(light_normal + view_normal)
                    specular_accumulator += light.specular * pow(max(0.0, np.dot(model_normal, halfway_vector)),
                                                                 intersection.specular_exponent)

        # Reflective Lighting
        reflection_color = np.zeros(3)
        if np.any(intersection.reflection_rgb):
            reflection_direction = self.d - np.dot(2*self.d, intersection.normal)*intersection.normal
            reflection_ray = Ray(eye=intersection.position, direction=reflection_direction)
            reflection_color = intersection.reflection_rgb * reflection_ray.propagate(scene, depth=depth+1)

        # Calculate Blinn-Phong Lighting Intensity
        return intersection.ambient + diffuse_accumulator + specular_accumulator + reflection_color


class Triangle:
    def __init__(self, p1, p2, p3):
        if not isinstance(p1, Point) or not isinstance(p2, Point) or not isinstance(p3, Point):
            raise TypeError("Input must be of type Point")

        self.__vertices = [p1, p2, p3]

    def intersects_ray(self, ray, bounds=None):
        """
        Calculates the ray-object intersection for triangles.
        The intersection distance must fall within the bounds.

        :param ray:
            (Ray) 3-D ray for intersection check

        :param bounds:
            (list) 2-element list containing the minimum and maximum t values

            Default: [0, inf]

        :return:
            (dict) Contains the intersection distance 't' and the intersection point with interpolated properties
        """
        if bounds is None:
            bounds = [0, inf]
        if len(bounds) != 2:
            raise ValueError("Bounds should be a 2-element array containing the lower and upper bounds")
        if not isinstance(ray, Ray):
            raise TypeError("Input must be of type Ray")

        # Ray-Triangle intersection from pg.79 of Fundamentals of Computer Graphics 4th Ed. by Marschner & Shirley
        a = self.vertices[0].x - self.vertices[1].x
        b = self.vertices[0].y - self.vertices[1].y
        c = self.vertices[0].z - self.vertices[1].z
        d = self.vertices[0].x - self.vertices[2].x
        e = self.vertices[0].y - self.vertices[2].y
        f = self.vertices[0].z - self.vertices[2].z
        g = ray.d[0]
        h = ray.d[1]
        i = ray.d[2]
        j = self.vertices[0].x - ray.e.x
        k = self.vertices[0].y - ray.e.y
        l = self.vertices[0].z - ray.e.z
        m = a * (e * i - h * f) + b * (g * f - d * i) + c * (d * h - e * g)

        t = -(f * (a * k - j * b) + e * (j * c - a * l) + d * (b * l - k * c)) / m
        if t < bounds[0] or t > bounds[1]:
            return None

        gamma = (i * (a * k - j * b) + h * (j * c - a * l) + g * (b * l - k * c)) / m
        if gamma < 0 or gamma > 1:
            return None

        beta = (j * (e * i - h * f) + k * (g * f - d * i) + l * (d * h - e * g)) / m
        if beta < 0 or beta > 1 - gamma:
            return None

        alpha = 1 - beta - gamma

        # Calculate color by barycentric interpolation of vertex colors
        intersection_point = Point(
            pos=np.array([
                alpha * self.vertices[0].x + beta * self.vertices[1].x + gamma * self.vertices[2].x,
                alpha * self.vertices[0].y + beta * self.vertices[1].y + gamma * self.vertices[2].y,
                alpha * self.vertices[0].z + beta * self.vertices[1].z + gamma * self.vertices[2].z
            ]),
            # pos=np.array(ray.e.position + t * ray.d),
            normals=np.array([
                alpha * self.vertices[0].nx + beta * self.vertices[1].nx + gamma * self.vertices[2].nx,
                alpha * self.vertices[0].ny + beta * self.vertices[1].ny + gamma * self.vertices[2].ny,
                alpha * self.vertices[0].nz + beta * self.vertices[1].nz + gamma * self.vertices[2].nz
            ]),
            rgba=np.array([
                alpha * self.vertices[0].r + beta * self.vertices[1].r + gamma * self.vertices[2].r,
                alpha * self.vertices[0].g + beta * self.vertices[1].g + gamma * self.vertices[2].g,
                alpha * self.vertices[0].b + beta * self.vertices[1].b + gamma * self.vertices[2].b,
                alpha * self.vertices[0].a + beta * self.vertices[1].a + gamma * self.vertices[2].a
            ]),
            reflection=np.array([
                alpha * self.vertices[0].reflection_r + beta * self.vertices[1].reflection_r + gamma * self.vertices[2].reflection_r,
                alpha * self.vertices[0].reflection_g + beta * self.vertices[1].reflection_g + gamma * self.vertices[2].reflection_g,
                alpha * self.vertices[0].reflection_b + beta * self.vertices[1].reflection_b + gamma * self.vertices[2].reflection_b,
                alpha * self.vertices[0].reflection_a + beta * self.vertices[1].reflection_a + gamma * self.vertices[2].reflection_a
            ])
        )
        return {'t': t, 'intersection': intersection_point}

    @property
    def vertices(self):
        return self.__vertices

    @vertices.setter
    def vertices(self, vertices):
        if len(vertices) != 3:
            raise ValueError("Triangles have only 3 vertices")
        for vertex in vertices:
            if not isinstance(vertex, Point):
                raise TypeError("Input must be of type Point")

        self.__vertices = vertices


class MeshIterator:
    def __init__(self, mesh):
        self.__mesh_list = mesh
        self.__index = 0

    def __next__(self):
        if self.__index < len(self.__mesh_list):
            self.__index += 1
            return self.__mesh_list[self.__index - 1]
        raise StopIteration


class Mesh:
    def __init__(self, name=None):
        self.__mesh = []
        self.__bounding_box = {'min': Point(),
                               'max': Point()}
        self.__name = name or ''.join(
            random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))

    def append(self, tri: Triangle) -> int:
        """
        Appends a Triangle to the mesh and returns the new length of the mesh.

        :param tri:
            (Triangle) triangle to append to the mesh

        :return:
            (int) length of mesh
        """

        if not isinstance(tri, Triangle):
            raise TypeError("Input must be of type Triangle")
        self.__mesh.append(tri)
        return len(self)

    def calculate_bounding_box(self):
        bb_min = Point(np.array([inf, inf, inf]))
        bb_max = Point(np.array([-inf, -inf, -inf]))
        for tri in self.__mesh:
            for vertex in tri.vertices:
                if vertex.x < bb_min.x:
                    bb_min.x = vertex.x
                if vertex.y < bb_min.y:
                    bb_min.y = vertex.y
                if vertex.z < bb_min.z:
                    bb_min.z = vertex.z
                if vertex.x > bb_max.x:
                    bb_max.x = vertex.x
                if vertex.y > bb_max.y:
                    bb_max.y = vertex.y
                if vertex.z > bb_max.z:
                    bb_max.z = vertex.z
        self.__bounding_box['min'] = bb_min
        self.__bounding_box['max'] = bb_max

    def bounding_box_intersects_ray(self, ray: Ray):
        """
        Calculates if a ray intersects the bounding box of the mesh.

        :param ray:
            The ray to test for intersection.

        :return:
            (bool) Intersection occurs
        """
        if not isinstance(ray, Ray):
            raise TypeError("Input must be of type Ray")
        pass

    def load_indexed_geometry(self, vertex_data: np.ndarray, index_data: np.ndarray, endpoints: np.ndarray,
                              transformation: np.ndarray, materials=None):
        """
        Generates a triangle mesh from data organized as indexed geometry.

        :param vertex_data:
            An interleaved array of floating point vertex data.

        :param index_data:
            An array of unsigned integers for the index/element buffer. Specifies how the triangles are connected.

        :param endpoints:
            A 2-element array containing the indices for the index_data array.

        :param transformation:
            An initial transformation to move the vertices into the coordinate space for processing. Generally, this
            moves from model space to camera (eye) space.

        :param materials:
            A dictionary mapping mesh names to lighting/color properties

        """
        if not isinstance(transformation, np.ndarray):
            raise TypeError("Transformation must be a numpy nd-array of size 4x4")
        if not transformation.shape == (4, 4):
            raise TypeError("Transformation must be a numpy nd-array of size 4x4")

        num_elements = 3 * (endpoints[-1] - endpoints[0])
        start_index = 3 * endpoints[0]
        end_index = start_index + num_elements

        build_tri = []
        vertex_indices = index_data[start_index:end_index]
        for vertex_index in vertex_indices:
            pos = np.matmul(transformation, np.array([vertex_data[vertex_index],
                                                      vertex_data[vertex_index + 1],
                                                      vertex_data[vertex_index + 2],
                                                      1])
                            )
            nrm = np.matmul(transformation, np.array([vertex_data[vertex_index + 3],
                                                      vertex_data[vertex_index + 4],
                                                      vertex_data[vertex_index + 5],
                                                      1])
                            )

            # TODO: Load color from texture mapping
            if materials:
                color = np.copy(materials[self.name]['color'])
                color = np.append(color, 1.0)
                build_tri.append(Point(pos=pos, normals=nrm, rgba=color))
            else:
                build_tri.append(Point(pos=pos, normals=nrm))

            # Create triangle for every 3 vertices processed
            if len(build_tri) == 3:
                self.append(Triangle(build_tri[0], build_tri[1], build_tri[2]))
                build_tri.clear()

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, val):
        self.__name = str(val)

    @property
    def bounding_box(self):
        return self.__bounding_box

    @property
    def mesh(self) -> list:
        return self.__mesh

    def __len__(self):
        return len(self.__mesh)

    def __getitem__(self, item):
        return self.__mesh[item]

    def __iter__(self):
        return MeshIterator(self.__mesh)
