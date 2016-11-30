###########
##### Setup
###########

Before our demos can be started, embree must be set up.
For this we refer to readme.pdf.
In addition to the standard embree requirements our system depends on the eigen3-library.

Our code has been tested and measurements have been taken on Ubuntu 14.04.



###########
##### Start
###########

To start a simple interactive demonstration for Ray Tracing of Compressed Representation for Parametric Surfaces simply run

$ ./viewer -c ./models/bomberman.ecs

in the embree root directory.

With the F1 and F4 keys it is possible to switch between simple shading and UV-representation.





###############
##### Box Types
###############

Different box types can be enabled by simply uncommenting the desired type in the ./models/bomberman.ecs file.
Possible box types and their parameter are:

1. Reference box with floating point precision:
-oriented.ref

2. Quantized boxes without further compression (pre == non-uniform quantization / uni == uniform quantization):
-oriented.pre332non
-oriented.uni332non

3. Quantized boxes with slab compression (pre == non-uniform quantization / uni == uniform quantization):
-oriented.pre332a
-oriented.uni332a

4. Quantized boxes with half-slab compression (pre == non-uniform quantization / uni == uniform quantization):
-oriented.pre332b
-oriented.uni332b





#####################
##### Further Options
#####################

You can find further options in the ./kernels/xeon/geometry/oriented.h file to adjust and test various aspects of our system.
These options can be enabled and disabled by commenting and uncommenting following preprocessor defines in the oriented.h file.
All four options are not exclusive an can be enabled in every possible permutation.

#define PROJECTION  (enabled by default)
#define ORTHONORMAL (disabled by default)
#define EARLY_OUT	(enabled by default)
#define WITH_GRID	(disabled by default)	


### PROJECTION
By enabling this option patches will be projected to a square shape thus allowing the system to 
faithfully represent trapezoid patches with axis aligned bounding boxes.

### ORTHONORMAL
By enabling this option the system uses an orthonormal system for the local reorientation of the patch.
If disabled, the system will use a sheered but still normalized coordinate system.
If projection is disabled a sheered coordinate system can deliver quite reasonable results.

### EARLY_OUT
If this option is enabled, ray traversal will exit the bottom level BVH after the first leaf intersection.
Since the traverser sorts intersected boxes, the first hit with geometry within a CBVH is also the nearest.

### WITH_GRID
With this option enabled the traversal does not stop with the intersection of the leaf-boxes, but 
tests for intersections with the exact geometry.
To do so the traverser reconstructs 2 triangles from the stored vertex grid.





########################
##### Subdivision Levels
########################

You can also define the subdivison level and the level where the transition to our CBVH shall happen inside
./kernels/xeon/geometry/oriented.h by adjusting the following options:

#define LEVEL_N 2
#define LEVEL_Q 3

### LEVEL_N
Subdivision levels for the top-level BVH.
Neither quantization nor compression is applied here.
Levels <2 are not working currently for the top-level BVH.

### LEVEL_Q
Subdivision levels for the bottom-level BVH.
Quantization and compression is applied here.
