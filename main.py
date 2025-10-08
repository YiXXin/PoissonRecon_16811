import numpy as np
import open3d as o3d
from random_adaptive_resampling import data_aug_rar
from utils import read_pcd_file, visualize_pcd
import mcubes
from py_psr import psr


if __name__ == '__main__':
    file_path = "test_data/dragon.ply"

    # Read files
    P, N = read_pcd_file(file_path)

    print("P.shape:", P.shape)
    print("N.shape:", N.shape)

    # Sparse Data
    # np.random.seed(0)
    # idx = np.random.choice(P.shape[0], P.shape[0] // 2, replace=False)
    # P_sample = P[idx, :]
    # N_sample = N[idx, :]
    # print("P_sample.shape:", P_sample.shape)
    # print("N_sample.shape:", N_sample.shape)

    # Noise Data
    # random_noise = np.random.normal(0, np.std(np.var(P, axis=0))*0.1, P.shape)
    # P += random_noise

    # # Visualize the original pcd
    # visualize_pcd(P, N)

    # Data augmentation
    # P_syn, N_syn = data_aug_rar(P, N, cube_n=16)
    # P = np.concatenate((P, P_syn), axis=0)
    # N = np.concatenate((N, N_syn), axis=0)
    # print("P.shape (after augmentation):", P.shape)
    # print("N.shape (after augmentation):", N.shape)

    # # Visualize the augmented pcd
    # visualize_pcd(P, N)

    cubes_n = 64

    file_name = file_path.split("/")[-1].split(".")[0]
    save_path = f"./output_data/cube_{cubes_n}_"+file_name+".obj"
    print("Poisson surface reconstruction...")
    vertices, triangles = psr(P, N, cubes_n=cubes_n, padding=8, cg_maxiter=2000, cg_tol=1e-5)
    print(save_path + " saved!")
    mcubes.export_obj(vertices, triangles, save_path)



