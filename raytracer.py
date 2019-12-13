import numpy as np
import json
import imageio
import random
from geometry import *
from geometry_generator import *
from lights import *
from scene import *
from math import inf, cos, sin

random.seed(2019)

AA_LEVEL = 2
IMAGE_SIZE = [240, 420]
view_volume = Volume()
camera_origin = Point([0, 0, 0])

# Load MINI model
# with open('mini_geometry.json', 'r') as json_file:
#     geometry_data = json.load(json_file)
# with open('mini_material.json', 'r') as json_file:
#     material_data = json.load(json_file)
with open('bounding_box.json', 'r') as json_file:
    geometry_data = json.load(json_file)
with open('bounding_material.json', 'r') as json_file:
    material_data = json.load(json_file)

rot = np.array([
    [1, 0, 0, 0],
    [0, cos(np.deg2rad(-90)), -sin(np.deg2rad(-90)), 0],
    [0, sin(np.deg2rad(-90)), cos(np.deg2rad(-90)), 0],
    [0, 0, 0, 1]
])
scl = np.array([
    [1 / 181 / 5, 0, 0, 0],
    [0, 1 / 181 / 5, 0, 0],
    [0, 0, 1 / 181 / 5, 0],
    [0, 0, 0, 1]
])
trn = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, -0.5],
    [0, 0, 0, 1]
])

model_transform = np.matmul(trn, np.matmul(scl, rot))

modelview_transform = np.array([
    [0.0011049723252654076, 0, 0, 0],
    [0, 0, -0.0011049723252654076, 0],
    [0, 0.0011049723252654076, 0, 0],
    [0, 0, 0, 1]
])

scene = Scene()

# Add MINI components
# for mini_component in geometry_data['groups']:
#     component = Mesh(name=mini_component)
#     component.load_indexed_geometry(geometry_data['vertexdata'], geometry_data['indexdata'],
#                                     geometry_data['groups'][mini_component], model_transform, material_data)
#     scene.add_geometry(component)


# Add reflective box
generate_boxes(scene)

# Create scene lights
scene.add_light(Light(position=[-1, -1, -1]))
# scene.add_light(Light(position=[1, 1, -1]))
# scene.add_light(Light(position=[0, 1, -1]))

# Ray tracer
rgb_size = np.copy(IMAGE_SIZE)
rgb_size = np.append(rgb_size, 3)
float64_img = np.zeros(rgb_size).astype(np.float64)

for i in range(0, IMAGE_SIZE[0]):
    for j in range(0, IMAGE_SIZE[1]):
        pixel_colors = []
        for aa_num in range(0, AA_LEVEL):
            if AA_LEVEL == 1:
                u = view_volume.left + (2 * (j + 0.5)) / IMAGE_SIZE[1]
                v = view_volume.bottom + (2 * (i + 0.5)) / IMAGE_SIZE[1]
            else:
                u = view_volume.left + (2 * (j + 0.5 + random.uniform(-0.49, 0.49))) / IMAGE_SIZE[1]
                v = view_volume.bottom + (2 * (i + 0.5 + random.uniform(-0.49, 0.49))) / IMAGE_SIZE[0]

            primary_ray = Ray(eye=camera_origin, direction=[u, v, -view_volume.near])
            pixel_colors.append(primary_ray.propagate(scene))

        pixel_colors = np.array(pixel_colors)
        float64_img[i, j, :] = np.average(pixel_colors, axis=0)

# uint8_img = np.divide(float64_img.astype(np.float64), float64_img.max(initial=-inf))
uint8_img = 255 * np.clip(float64_img, 0, 1)
uint8_img = uint8_img.astype(np.uint8)

imageio.imwrite('output.png', np.flipud(uint8_img))