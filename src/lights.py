from geometry import *


class Light:
    def __init__(self, position=np.ones(3), target=np.zeros(3), diffuse_color=np.ones(3), specular_color=np.ones(3)):
        if isinstance(position, list):
            if not len(position) == 3:
                raise ValueError("Position should be a length-3 vector")
            position = np.array(position)
        elif isinstance(position, np.ndarray):
            if not np.size(position) == 3:
                raise ValueError("Position should be a length-3 vector")
        else:
            raise TypeError("Unexpected type for position")

        if isinstance(target, list):
            if not len(target) == 3:
                raise ValueError("Direction should be a length-3 vector")
            target = np.array(target)
        elif isinstance(target, np.ndarray):
            if not np.size(target) == 3:
                raise ValueError("Direction should be a length-3 vector")
        else:
            raise TypeError("Unexpected type for direction")

        self.__position = Point(pos=position, normals=target-position)

        if isinstance(diffuse_color, list):
            if not len(diffuse_color) == 3:
                raise ValueError("Diffuse color should be a length-3 vector")
            self.__diffuse = np.array(diffuse_color)
        elif isinstance(diffuse_color, np.ndarray):
            if not np.size(diffuse_color) == 3:
                raise ValueError("Diffuse color should be a length-3 vector")
            self.__diffuse = diffuse_color

        if isinstance(specular_color, list):
            if not len(specular_color) == 3:
                raise ValueError("Specular color should be a length-3 vector")
            self.__specular = np.array(specular_color)
        elif isinstance(specular_color, np.ndarray):
            if not np.size(specular_color) == 3:
                raise ValueError("Diffuse color should be a length-3 vector")
            self.__specular = specular_color

    @property
    def position(self):
        return self.__position.position

    @property
    def diffuse(self):
        return np.array(self.__diffuse)

    @property
    def specular(self):
        return np.array(self.__specular)
