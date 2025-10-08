import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.linalg import cg
import mcubes
import open3d as o3d
import time


def matrix_L(cubes_n, cubes_step):
    grid_num = cubes_n ** 3
    grid_idx = np.arange(grid_num).reshape((cubes_n, cubes_n, cubes_n))

    staggered_grid_num = (cubes_n - 1) * (cubes_n ** 2)

    data_rows = np.tile(np.arange(staggered_grid_num), 2)
    data_cols_x = np.hstack((grid_idx[1:, :, :].reshape(-1), grid_idx[:-1, :, :].reshape(-1)))
    data_cols_y = np.hstack((grid_idx[:, 1:, :].reshape(-1), grid_idx[:, :-1, :].reshape(-1)))
    data_cols_z = np.hstack((grid_idx[:, :, 1:].reshape(-1), grid_idx[:, :, :-1].reshape(-1)))

    data_x = np.array([1 / cubes_step[0]] * staggered_grid_num + [-1 / cubes_step[0]] * staggered_grid_num)
    data_y = np.array([1 / cubes_step[1]] * staggered_grid_num + [-1 / cubes_step[1]] * staggered_grid_num)
    data_z = np.array([1 / cubes_step[2]] * staggered_grid_num + [-1 / cubes_step[2]] * staggered_grid_num)

    L_x = coo_matrix((data_x, (data_rows, data_cols_x)), shape=(staggered_grid_num, grid_num))
    L_y = coo_matrix((data_y, (data_rows, data_cols_y)), shape=(staggered_grid_num, grid_num))
    L_z = coo_matrix((data_z, (data_rows, data_cols_z)), shape=(staggered_grid_num, grid_num))

    L = vstack((L_x, L_y, L_z))

    return L


def weights_alpha(cubes_n, cubes_step, bottom_left, P, direction):
    if direction == "x":
        bottom_left[0] += 0.5 * cubes_step[0]
    elif direction == "y":
        bottom_left[1] += 0.5 * cubes_step[1]
    elif direction == "z":
        bottom_left[2] += 0.5 * cubes_step[2]

    bottom_left_x, bottom_left_y, bottom_left_z = bottom_left[0], bottom_left[1], bottom_left[2]

    grid_bottom_left = np.vstack((
        (P[:, 0] - bottom_left_x) // cubes_step[0],
        (P[:, 1] - bottom_left_y) // cubes_step[1],
        (P[:, 2] - bottom_left_z) // cubes_step[2]
    )).astype(int)

    grid_coord = np.vstack((
        (P[:, 0] - bottom_left_x) / cubes_step[0] - grid_bottom_left[0],
        (P[:, 1] - bottom_left_y) / cubes_step[1] - grid_bottom_left[1],
        (P[:, 2] - bottom_left_z) / cubes_step[2] - grid_bottom_left[2]
    ))

    alpha = np.hstack((
        (1 - grid_coord[0]) * (1 - grid_coord[1]) * (1 - grid_coord[2]),
        grid_coord[0] * (1 - grid_coord[1]) * (1 - grid_coord[2]),
        (1 - grid_coord[0]) * grid_coord[1] * (1 - grid_coord[2]),
        grid_coord[0] * grid_coord[1] * (1 - grid_coord[2]),
        (1 - grid_coord[0]) * (1 - grid_coord[1]) * grid_coord[2],
        grid_coord[0] * (1 - grid_coord[1]) * grid_coord[2],
        (1 - grid_coord[0]) * grid_coord[1] * grid_coord[2],
        grid_coord[0] * grid_coord[1] * grid_coord[2]
    ))


    if direction == "x" or direction == "y" or direction == "z":
        grid_num = (cubes_n - 1) * (cubes_n ** 2)
    else:
        grid_num = cubes_n ** 3

    if direction == "x":
        staggered_grid_idx = np.arange(grid_num).reshape((cubes_n - 1, cubes_n, cubes_n))
    elif direction == "y":
        staggered_grid_idx = np.arange(grid_num).reshape((cubes_n, cubes_n - 1, cubes_n))
    elif direction == "z":
        staggered_grid_idx = np.arange(grid_num).reshape((cubes_n, cubes_n, cubes_n - 1))
    else:
        staggered_grid_idx = np.arange(grid_num).reshape((cubes_n, cubes_n, cubes_n))

    data_rows = np.tile(np.arange(P.shape[0]), 8)

    data_cols = np.hstack((
        staggered_grid_idx[grid_bottom_left[0], grid_bottom_left[1], grid_bottom_left[2]],
        staggered_grid_idx[grid_bottom_left[0] + 1, grid_bottom_left[1], grid_bottom_left[2]],
        staggered_grid_idx[grid_bottom_left[0], grid_bottom_left[1] + 1, grid_bottom_left[2]],
        staggered_grid_idx[grid_bottom_left[0] + 1, grid_bottom_left[1] + 1, grid_bottom_left[2]],
        staggered_grid_idx[grid_bottom_left[0], grid_bottom_left[1], grid_bottom_left[2] + 1],
        staggered_grid_idx[grid_bottom_left[0] + 1, grid_bottom_left[1], grid_bottom_left[2] + 1],
        staggered_grid_idx[grid_bottom_left[0], grid_bottom_left[1] + 1, grid_bottom_left[2] + 1],
        staggered_grid_idx[grid_bottom_left[0] + 1, grid_bottom_left[1] + 1, grid_bottom_left[2] + 1],
    ))

    weights = coo_matrix((alpha, (data_rows, data_cols)), shape=(P.shape[0], grid_num))
    return weights


def psr(P, N, cubes_n, padding, cg_maxiter=2000, cg_tol=1e-5):
    box_size = np.max(P, 0) - np.min(P, 0)
    cubes_step = box_size / cubes_n
    bottom_left = np.min(P, 0) - padding * cubes_step
    cubes_n += 2 * padding

    L = matrix_L(cubes_n, cubes_step)

    weight_x = weights_alpha(cubes_n, cubes_step, bottom_left, P, "x")
    weight_y = weights_alpha(cubes_n, cubes_step, bottom_left, P, "y")
    weight_z = weights_alpha(cubes_n, cubes_step, bottom_left, P, "z")
    weight = weights_alpha(cubes_n, cubes_step, bottom_left, P, "None")

    v = np.hstack((
        weight_x.T @ N[:, 0],
        weight_y.T @ N[:, 1],
        weight_z.T @ N[:, 2]
    ))

    print("solving...")
    tic = time.time()
    x, _ = cg(L.T @ L, L.T @ v, maxiter=cg_maxiter, tol=cg_tol)
    toc = time.time()
    print("time:", toc - tic)
    print("solved!")

    sigma = np.mean(weight @ x)
    x -= sigma
    x = x.reshape(cubes_n, cubes_n, cubes_n)
    vertices, triangles = mcubes.marching_cubes(x, 0)
    vertices[:, 0] = vertices[:, 0] * cubes_step[0] + bottom_left[0]
    vertices[:, 1] = vertices[:, 1] * cubes_step[1] + bottom_left[1]
    vertices[:, 2] = vertices[:, 2] * cubes_step[2] + bottom_left[2]

    print("drawing...")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    return vertices, triangles

