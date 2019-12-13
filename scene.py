from geometry import *
from lights import *
import numpy as np
from math import inf


class Scene:
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

    def trace(self, ray) -> tuple:
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
