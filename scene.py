from geometry import *
from lights import *
import numpy as np
from math import inf


class Scene:
    """
    Object to hold the geometry and lights of a 3-D scene and trace rays to find the minimal point of intersection.
    """
    acne_offset = 0.00000001

    def __init__(self):
        self.__geometry = []
        self.__lights = []

    def add_geometry(self, mesh):
        if not isinstance(mesh, Mesh):
            raise TypeError("Expected Mesh component")
        self.__geometry.append(mesh)

    def add_light(self, light):
        if not isinstance(light, Light):
            raise TypeError("Expected Light component")
        self.__lights.append(light)

    def trace(self, ray: Ray) -> tuple:
        """
        Traces a given ray through the scene to find the minimal point of intersection.

        :param ray:
            (Ray) The ray propagating through the scene

        :return:
            (tuple) A set containing the offset 't' found for minimal intersection and the intersection point
                    with interpolated values.
        """
        if not isinstance(ray, Ray):
            raise TypeError("Trace only works with class Ray")

        t_min = inf
        intersection_point = Point()
        for component in self.geometry:
            # TODO: Test bounding box intersection

            for tri in component:
                intersection_info = tri.intersects_ray(ray, bounds=[Scene.acne_offset, t_min])
                if intersection_info is not None:
                    if intersection_info['t'] < t_min:
                        t_min = intersection_info['t']
                        intersection_point = intersection_info['intersection']

        return t_min, Point() if t_min == inf else intersection_point

    @property
    def geometry(self):
        return self.__geometry

    @property
    def lights(self):
        return self.__lights
