# PoissonRecon-in-Python

### Poisson Problem

The surface separates the object itself and its surrounding environment.
Thus, we can define an implicit function such that its values inside and
outside the object are 1 and 0 respectively, and its boundary,
delineating between 1 and 0, can easily be defined as the surface. We
denote such a 3D indicator function as $\chi$, with its value at any
position $p$ being $\chi(p)$. Solving this function is the core step of
Poisson surface reconstruction.

We currently know that the vector field of the normals to the sample
point clouds on the actual surface, denoted as $\vec{V}$, which is
perpendicular to the surface with unit length. Since $\chi$ also
conforms to these properties according to its definition, in the limited
continuous space $\Omega$ containing all the sample points
$p \in \Omega$, we aim for the gradient of the scalar function $\chi$ to
be as close as possible to $\vec{V}$, i.e.:
$$\min_\chi{\int_{\Omega}|\nabla \chi-\vec{V}|^2 dp}$$

Applying the Euler-Lagrange Equation to
$F(p, \chi(p), \nabla \chi(p))= |\nabla \chi-\vec{V}|^2$, we obtain:
$$F_{\chi}-\frac{\partial}{\partial p} F_{\nabla \chi}=0$$

Since $F_{\chi}=0$, $F_{\nabla \chi}=2(\nabla \chi-\vec{V})$, and
$\frac{\partial}{\partial p} F_{\nabla \chi}=2(\nabla \cdot \nabla \chi - \nabla \vec{V})$,
the problem transforms into a standard Poisson problem:
$$\nabla \cdot \nabla \chi \equiv \Delta \chi = \nabla \vec{V}$$ where
$\nabla$ represents the gradient operator and $\Delta$ represents the
Laplacian operator. $\Delta \chi$ signifies its divergence.

### Space Discretization

Computer-implemented surface reconstruction is executed in discrete
space. Hence, $\Omega$ can be conceptualized as a cuboid area. This
area's bottom left and upper right corner points are determined by the
minimum and maximum values along the three dimensions of all limited
sample point clouds' coordinates, respectively. This area is then
subdivided evenly into $n$ intervals along three axes, forming $n^3$
grids as depicted in Figure 1(a). Similarly, we define a scalar function
$\mathbf{x}$, with an isovalue of $\mathbf{x}(q_i)$ for any $q_i$ with
$i \in \mathbb{Z} \cap [1, n^3]$.

<div align="center">
  <img src="https://github.com/Kaihua-Chen/PoissonRecon-in-Python/tree/main/figures/figure1.png">
</div>

The gradient of $\mathbf{x}$ necessitates discrete gradient calculation.
To this end, we utilize staggered grids, offset by $\frac{1}{2}$ grid
along x, y, or z axis, as shown in Figure 1(b). The gradient of any point on the staggered
grids $p_{i-\frac{1}{2}}$ in a particular direction can be calculated
using its adjacent original grids $p_i$ and $p_{i-1}$ as follows:
$$\frac{d \mathbf{x}(p_{i-\frac{1}{2}})}{dp}= \frac{\mathbf{x}(p_i)-\mathbf{x}(p_{i-1})}{p_i-p_{i-1}}$$

This calculation can be performed by left-multiplying with a matrix
$\mathbf{L}$.

Furthermore, since the sample points $\mathbf{s}$ are often not located
precisely on the grids, we employ trilinear interpolation on their
normals $\mathbf{s}.\vec{N}$ within their resoective subgrid to
approximate the vector field $\mathbf{v}$ on the staggered grids.
Consequently, our objective becomes:
$$\min_{\mathbf{x}} {||\mathbf{Lx}-\mathbf{v}||^2}$$

The least squares solution for this can be expressed as:
$$\mathbf{L^T L x} = \mathbf{L^T v}$$

This equation bears resemblance to the Poisson problem, where
$\mathbf{L^T L}$ on the left side represents $\Delta$ and $\mathbf{L^T}$
on the right side represents $\nabla$.

Since $\mathbf{L}$ is a large sparse matrix, we opt for solving this
equation through conjugate gradient descent instead of its closed-form
solution, reducing the computational complexity. After computing
$\mathbf{x}$ and normalizing it, the marching cubes
algorithm is directly employed for surface
reconstruction.

### Random Adaptive Resampling

While highly non-uniform point clouds are relatively uncommon, they can
significantly impact the quality of surface reconstruction. To address
this issue, Michael Kazhadan *et al*. proposed the adaptive grid method.
This method increases the weights of the vector field of sparsely
sampled points by estimating their local sampling density.

Since this method requires accurate evaluation of the density for all
points and the assignment of corresponding weights, it entails
relatively large computational complexity. Building on the understanding
that the order of samples does not affect the surface reconstruction
results, we achieve similar effects by augmenting the sample numbers
through Random Adaptive Resampling (RAR) in the preprocessing stage.

Specifically, for Poisson surface reconstruction with a resolution of
$n^3$, we calculate the number of samples $k_i$ within each subgrid at a
lower resolution of $(n-2)^3$, where
$i \in \mathbb{Z} \cap [1, (n-2)^3]$. For all subgrids with $k_i>0$, we
randomly resample them until their counts equal $\max_{i}{k_i}$. The
effectiveness and robustness of RAR can be tested using "horse_with_normals.xyz" example

### Python Implementation

We use *Python* to implement the classical surface reconstruction
algorithm from scratch. Our implementation relies on essential
libraries such as *numpy* and *open3d*. Open3d serves as a tool for
reading and visualizing the 3D point cloud data.

Upon loading $N$ sample points, the dimensions of their coordinates and
normals are both $(N, 3)$. Assuming the number of grids after padding is
$n^3$, the dimensions of the gradient operators $\mathbf{L}_x$,
$\mathbf{L}_y$, and $\mathbf{L}_z$ are each $((n-1)n^2, n^3)$.
Consequently, the dimension of
$\mathbf{L}=(\mathbf{L}_x, \mathbf{L}_y, \mathbf{L}_z)^T$ becomes
$(3(n-1)n^2, n^3)$. Similarly, the dimensions of the weights for normals
along the three dimensions after trilinear interpolation,
$\mathbf{w}_x$, $\mathbf{w}_y$, and $\mathbf{w}_z$ are each
$(N, (n-1)n ^2)$. The dimension of the vector field
$\mathbf{v}=(\mathbf{w}_x^T \cdot \mathbf{s}.\vec{N}_x, \mathbf{w}_y^T \cdot \mathbf{s}.\vec{N}_y, \mathbf{w}_z^T \cdot \mathbf{s}.\vec{N}_z)^T$
is $(3(n-1)n^2, 1)$. Thus, the dimension of $\mathbf{x}$, as resolved
using *scipy.sparse.linalg.cg*, is $(n^3, 1)$.

Additionally, $\mathbf{x}$ can be normalized so that it is bounded by 0.
Following a similar process to the derivation of other weight matrices,
the dimension of the global weight $\mathbf{w}$ is $(N, n^3)$. We then
compute $\sigma = \frac{1}{N} \mathbf{1}^T \mathbf{w} \cdot \mathbf{x}$.
Finally, the surface reconstruction is completed by subtracting $\sigma$
from $\mathbf{x}$ and applying *mcubes.marching_cubes($\mathbf{x}$, 0)*.


### References
[1] Kazhdan, M., Bolitho, M., & Hoppe, H. (2006, June). Poisson surface reconstruction. In Proceedings of the fourth Eurographics symposium on Geometry processing (Vol. 7, p. 0).

[2] Lorensen, W. E., & Cline, H. E. (1998). Marching cubes: A high resolution 3D surface construction algorithm. In Seminal graphics: pioneering efforts that shaped the field (pp. 347-353).

