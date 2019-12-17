from geometry import *
from math import cos, sin


def generate_box_scene(scene):
    # Cornell Box
    cube_face = Mesh(name="Bottom")
    cube_face.append(Triangle(Point(pos=[-1, -0.5, 0], normals=[-1, 1, 0], rgba=[0.94, 0.94, 0.94, 1],
                                    reflection=[1, 1, 1, 1]),
                              Point(pos=[1, -0.5, -1], normals=[1, 1, -1], rgba=[0.94, 0.94, 0.94, 1],
                                    reflection=[1, 1, 1, 1]),
                              Point(pos=[-1, -0.5, -1], normals=[-1, 1, -1], rgba=[0.94, 0.94, 0.94, 1],
                                    reflection=[1, 1, 1, 1])
                              ))
    cube_face.append(Triangle(Point(pos=[-1, -0.5, 0], normals=[-1, 1, 0], rgba=[0.94, 0.94, 0.94, 1],
                                    reflection=[1, 1, 1, 1]),
                              Point(pos=[1, -0.5, 0], normals=[1, 1, 0], rgba=[0.94, 0.94, 0.94, 1],
                                    reflection=[1, 1, 1, 1]),
                              Point(pos=[1, -0.5, -1], normals=[1, 1, -1], rgba=[0.94, 0.94, 0.94, 1],
                                    reflection=[1, 1, 1, 1])
                              ))
    scene.add_geometry(cube_face)

    cube_face = Mesh(name="Top")
    cube_face.append(Triangle(Point(pos=[-1, 0.5, 0], normals=[-1, -1, 0], rgba=[0.94, 0.94, 0.94, 1]),
                              Point(pos=[1, 0.5, -1], normals=[1, -1, -1], rgba=[0.94, 0.94, 0.94, 1]),
                              Point(pos=[-1, 0.5, -1], normals=[-1, -1, -1], rgba=[0.94, 0.94, 0.94, 1])
                              ))
    cube_face.append(Triangle(Point(pos=[-1, 0.5, 0], normals=[-1, -1, 0], rgba=[0.94, 0.94, 0.94, 1]),
                              Point(pos=[1, 0.5, 0], normals=[1, -1, 0], rgba=[0.94, 0.94, 0.94, 1]),
                              Point(pos=[1, 0.5, -1], normals=[1, -1, -1], rgba=[0.94, 0.94, 0.94, 1])
                              ))
    scene.add_geometry(cube_face)

    cube_face = Mesh(name="Back")
    cube_face.append(
        Triangle(Point(pos=[-1, -1, -0.15], normals=[-1, -1, 1], rgba=[0.94, 0.94, 0.94, 1], reflection=np.full(4, 1)),
                 Point(pos=[1, -1, -0.15], normals=[1, -1, 1], rgba=[0.94, 0.94, 0.94, 1], reflection=np.full(4, 1)),
                 Point(pos=[1, 1, -0.15], normals=[1, 1, 1], rgba=[0.94, 0.94, 0.94, 1], reflection=np.full(4, 1))
                 ))
    cube_face.append(
        Triangle(Point(pos=[-1, -1, -0.15], normals=[-1, -1, 1], rgba=[0.94, 0.94, 0.94, 1], reflection=np.full(4, 1)),
                 Point(pos=[1, 1, -0.15], normals=[1, 1, 1], rgba=[0.94, 0.94, 0.94, 1], reflection=np.full(4, 1)),
                 Point(pos=[-1, 1, -0.15], normals=[-1, 1, 1], rgba=[0.94, 0.94, 0.94, 1], reflection=np.full(4, 1))
                 ))
    scene.add_geometry(cube_face)

    cube_face = Mesh(name="Left")
    cube_face.append(Triangle(
        Point(pos=[-0.5, -0.5, 0], normals=[1, -0.5, 0], rgba=[1, 0, 0, 1], reflection=np.full(4, 0.4)),
        Point(pos=[-0.5, 0.5, -0.5], normals=[1, 0.5, -0.5], rgba=[1, 0, 0, 1], reflection=np.full(4, 0.4)),
        Point(pos=[-0.5, 0.5, 0], normals=[1, 0.5, 0], rgba=[1, 0, 0, 1], reflection=np.full(4, 0.4))
    ))
    cube_face.append(Triangle(
        Point(pos=[-0.5, -0.5, 0], normals=[1, -0.5, 0], rgba=[1, 0, 0, 1], reflection=np.full(4, 0.4)),
        Point(pos=[-0.5, -0.5, -0.5], normals=[1, -0.5, -0.5], rgba=[1, 0, 0, 1], reflection=np.full(4, 0.4)),
        Point(pos=[-0.5, 0.5, -0.5], normals=[1, 0.5, -0.5], rgba=[1, 0, 0, 1], reflection=np.full(4, 0.4))
    ))
    scene.add_geometry(cube_face)

    cube_face = Mesh(name="Right")
    cube_face.append(Triangle(Point(pos=[0.5, -0.5, 0], normals=[-1, -0.5, 0], rgba=[0 / 255, 255 / 255, 0 / 255, 1]),
                              Point(pos=[0.5, 0.5, -0.5], normals=[-1, 0.5, -0.5],
                                    rgba=[0 / 255, 255 / 255, 0 / 255, 1]),
                              Point(pos=[0.5, 0.5, 0], normals=[-1, 0.5, 0], rgba=[0 / 255, 255 / 255, 0 / 255, 1])
                              ))
    cube_face.append(Triangle(Point(pos=[0.5, -0.5, 0], normals=[-1, -0.5, 0], rgba=[0 / 255, 255 / 255, 0 / 255, 1]),
                              Point(pos=[0.5, -0.5, -0.5], normals=[-1, -0.5, -0.5],
                                    rgba=[0 / 255, 255 / 255, 0 / 255, 1]),
                              Point(pos=[0.5, 0.5, -0.5], normals=[-1, 0.5, -0.5],
                                    rgba=[0 / 255, 255 / 255, 0 / 255, 1])
                              ))
    scene.add_geometry(cube_face)

    # Mini Box
    # rot = np.array([
    #     [cos(np.deg2rad(45)), 0, sin(np.deg2rad(45)), 0],
    #     [0, 1, 0, 0],
    #     [-sin(np.deg2rad(45)), 0, cos(np.deg2rad(45)), 0],
    #     [0, 0, 0, 1]
    # ])
    rot = np.identity(4)
    cube_face = Mesh(name="Mini Bottom")
    cube_face.append(Triangle(
        Point(pos=[-0.15, -0.4, -0.075], normals=[-1, 1, 0], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[0.15, -0.4, -0.125], normals=[1, 1, -1], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[-0.15, -0.4, -0.125], normals=[-1, 1, -1], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot)
    ))
    cube_face.append(Triangle(
        Point(pos=[-0.15, -0.4, -0.075], normals=[-1, 1, 0], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[0.15, -0.4, -0.075], normals=[1, 1, 0], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[0.15, -0.4, -0.125], normals=[1, 1, -1], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot)
    ))
    scene.add_geometry(cube_face)

    cube_face = Mesh(name="Mini Top")
    cube_face.append(Triangle(
        Point(pos=[-0.15, 0, -0.075], normals=[-1, 1, 0], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[0.15, 0, -0.125], normals=[1, 1, -1], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[-0.15, 0, -0.125], normals=[-1, 1, -1], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot)
    ))
    cube_face.append(Triangle(
        Point(pos=[-0.15, 0, -0.075], normals=[-1, 1, 0], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[0.15, 0, -0.075], normals=[1, 1, 0], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[0.15, 0, -0.125], normals=[1, 1, -1], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot)
    ))
    scene.add_geometry(cube_face)

    cube_face = Mesh(name="Mini Back")
    cube_face.append(
        Triangle(
            Point(pos=[-0.15, -0.4, -0.125], normals=[-1, -1, 1], rgba=[0.94, 0.94, 0.94, 1],
                  reflection=np.full(4, 0.7)).transform(rot),
            Point(pos=[0.15, -0.4, -0.125], normals=[1, -1, 1], rgba=[0.94, 0.94, 0.94, 1],
                  reflection=np.full(4, 0.7)).transform(rot),
            Point(pos=[0.15, 0, -0.125], normals=[1, 1, 1], rgba=[0.94, 0.94, 0.94, 1],
                  reflection=np.full(4, 0.7)).transform(rot)
        ))
    cube_face.append(
        Triangle(
            Point(pos=[-0.15, -0.4, -0.125], normals=[-1, -1, 1], rgba=[0.94, 0.94, 0.94, 1],
                  reflection=np.full(4, 0.7)).transform(rot),
            Point(pos=[0.15, 0, -0.125], normals=[1, 1, 1], rgba=[0.94, 0.94, 0.94, 1],
                  reflection=np.full(4, 0.7)).transform(rot),
            Point(pos=[-0.15, 0, -0.125], normals=[-1, 1, 1], rgba=[0.94, 0.94, 0.94, 1],
                  reflection=np.full(4, 0.7)).transform(rot)
        ))
    scene.add_geometry(cube_face)

    cube_face = Mesh(name="Mini Front")
    cube_face.append(
        Triangle(
            Point(pos=[-0.15, -0.4, -0.075], normals=[-1, -1, 1], rgba=[0.94, 0.94, 0.94, 1],
                  reflection=np.full(4, 0.7)).transform(rot),
            Point(pos=[0.15, -0.4, -0.075], normals=[1, -1, 1], rgba=[0.94, 0.94, 0.94, 1],
                  reflection=np.full(4, 0.7)).transform(rot),
            Point(pos=[0.15, 0, -0.075], normals=[1, 1, 1], rgba=[0.94, 0.94, 0.94, 1],
                  reflection=np.full(4, 0.7)).transform(rot)
        ))
    cube_face.append(
        Triangle(
            Point(pos=[-0.15, -0.4, -0.075], normals=[-1, -1, 1], rgba=[0.94, 0.94, 0.94, 1],
                  reflection=np.full(4, 0.7)).transform(rot),
            Point(pos=[0.15, 0, -0.075], normals=[1, 1, 1], rgba=[0.94, 0.94, 0.94, 1],
                  reflection=np.full(4, 0.7)).transform(rot),
            Point(pos=[-0.15, 0, -0.075], normals=[-1, 1, 1], rgba=[0.94, 0.94, 0.94, 1],
                  reflection=np.full(4, 0.7)).transform(rot)
        ))
    scene.add_geometry(cube_face)

    cube_face = Mesh(name="Mini Left")
    cube_face.append(Triangle(
        Point(pos=[-0.15, -0.4, -0.075], normals=[1, -0.5, 0], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[-0.15, 0, -0.125], normals=[1, 0.5, -0.5], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[-0.15, 0, -0.075], normals=[1, 0.5, 0], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot)
    ))
    cube_face.append(Triangle(
        Point(pos=[-0.15, -0.4, -0.075], normals=[1, -0.5, 0], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[-0.15, -0.4, -0.125], normals=[1, -0.5, -0.5], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[-0.15, 0, -0.125], normals=[1, 0.5, -0.5], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot)
    ))
    scene.add_geometry(cube_face)

    cube_face = Mesh(name="Mini Right")
    cube_face.append(Triangle(
        Point(pos=[0.15, -0.4, -0.075], normals=[-1, -0.5, 0], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[0.15, 0, -0.125], normals=[-1, 0.5, -0.5], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[0.15, 0, -0.075], normals=[-1, 0.5, 0], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot)
    ))
    cube_face.append(Triangle(
        Point(pos=[0.15, -0.4, -0.075], normals=[-1, -0.5, 0], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[0.15, -0.4, -0.125], normals=[-1, -0.5, -0.5], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot),
        Point(pos=[0.15, 0, -0.125], normals=[-1, 0.5, -0.5], rgba=[0.94, 0.94, 0.94, 1],
              reflection=np.full(4, 0.7)).transform(rot)
    ))
    scene.add_geometry(cube_face)
