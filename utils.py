import open3d as o3d
import numpy as np

def read_pcd_file(file_path):
    if file_path.split(".")[-1] == "xyz":
        pcd = np.loadtxt(file_path)
        positions = pcd[:, :3]
        normals = pcd[:, 3:]
    elif file_path.split(".")[-1] == "ply":
        pcd = o3d.io.read_point_cloud(file_path)
        positions = np.asarray(pcd.points)
        # pcd.estimate_normals()
        normals = np.asarray(pcd.normals)

    return positions, normals


def visualize_pcd(P, N):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.normals = o3d.utility.Vector3dVector(N)
    o3d.visualization.draw_geometries([pcd])


