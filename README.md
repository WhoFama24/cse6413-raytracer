# cse6413-raytracer
Python-based raytracer for final project in the Mississippi State University CSE 6413 course.

This project demonstrates a simple ray tracer for the Cornell box.
The ray tracer does not implement acceleration structures or multiprocessing.

## Details
The Blinn-Phong lighting model is a foundational lighting model in computer graphics.
The output of this ray tracer using the standard Blinn-Phong model (ambient, diffuse, specular) is shown below.

![Blinn-Phong Ray Traced](/doc/img/box-16xAA-directional-phong.png?raw=true "Blinn-Phong Lighting")

This model can be further improved by using the inherent properties of ray tracing to add shadows and reflections.
The first image below shows the Blinn-Phong model with shadows, and the second image shows the Blinn-Phong model with shadows and reflections.

![Blinn-Phong Shadows Ray Traced](/doc/img/box-16xAA-directional-phong-shadows.png?raw=true "Blinn-Phong Lighting with Shadows")

![Blinn-Phong Shadows/Reflections Ray Traced](/doc/img/box-16xAA-directional-phong-shadows-reflections.png?raw=true "Blinn-Phong Lighting with Shadows and Reflections")

Additionally, the ray tracer can handle multiple lights in the scene.
The following image shows the same scene rendered with three directional lights instead of one.
This image demonstrates the hard shadow effect caused by ray tracing.

![Multi-Light Rendering](/doc/img/box-16xAA-3xML-phong-shadows-reflections.png?raw=true "Multi-Light Rendering")

Finally, a rendering was done with a purely reflective cube located inside the Cornell box.

![Reflective Cube Rendering](/doc/img/box-16xAA-1xML-multibox-full.png?raw=true "Reflective Box Rendering")

## Run
The scene geometry is hard coded into the program, and it currently does not support loading different scene files.

The ray tracer can simply be run by calling the `run.py` or `raytracer.py` file using a Python 3.7+ interpreter.
 - `run.py` executes the raytracer and times the execution time.
 - `raytracer.py` simply executes the program.
