import numpy as np
import open3d as o3d

np.random.seed(1234)

def data_aug_rar(positions, normals, cube_n=16):

    P = positions
    N = normals

    Cube_max = np.max(P, 0)
    Cube_min = np.min(P, 0)
    Cube_size = Cube_max - Cube_min

    cubes_n = cube_n
    cubes_step = Cube_size/cubes_n
    cubes_pos_dict = {}
    cubes_normal_dict = {}
    for i in range(cubes_n):
        for j in range(cubes_n):
            for k in range(cubes_n):
                tmp_pos = str(i)+","+str(j)+","+str(k)
                cubes_pos_dict[tmp_pos] = []
                cubes_normal_dict[tmp_pos] = []


    for idx in range(len(P)):
        i = int((P[idx,0]-Cube_min[0]) / cubes_step[0])
        j = int((P[idx,1]-Cube_min[1]) / cubes_step[1])
        k = int((P[idx,2]-Cube_min[2]) / cubes_step[2])
        if i == cubes_n:
            i -= 1
        if j == cubes_n:
            j -= 1
        if k == cubes_n:
            k -= 1
        tmp_pos = str(i) + "," + str(j) + "," + str(k)
        cubes_pos_dict[tmp_pos].append(P[idx])
        cubes_normal_dict[tmp_pos].append(N[idx])

    cubes_num_pts = []
    for key in cubes_pos_dict:
        cubes_num_pts.append(len(cubes_pos_dict[key]))
    cubes_num_pts.sort()
    cubes_num_pts = np.array(cubes_num_pts)
    cubes_num_pts = cubes_num_pts[np.where(cubes_num_pts>0)]

    down_percent = 0
    up_percent = 100
    local_aug_ratio = 1

    cubes_num_pts_p_down = np.percentile(cubes_num_pts, down_percent)
    cubes_num_pts_p_up = np.percentile(cubes_num_pts, up_percent)

    P_syn = []
    N_syn = []

    # flag = 4
    for key in cubes_pos_dict:
        if len(cubes_pos_dict[key]) >= cubes_num_pts_p_down and len(cubes_pos_dict[key])<cubes_num_pts_p_up:
            tmp_pos = key.split(",")
            i, j, k = int(tmp_pos[0]), int(tmp_pos[1]), int(tmp_pos[2])
            P_orig = np.array(cubes_pos_dict[key])
            N_orig = np.array(cubes_normal_dict[key])
            num_syn = int((cubes_num_pts_p_up - len(cubes_pos_dict[key]))*local_aug_ratio)

            for idxx in range(num_syn):
                rand_ind = np.random.randint(len(P_orig))
                P_syn.append([P_orig[rand_ind, 0], P_orig[rand_ind, 1], P_orig[rand_ind, 2]])
                N_syn.append([N_orig[rand_ind, 0], N_orig[rand_ind, 1], N_orig[rand_ind, 2]])

    P_syn = np.array(P_syn)
    N_syn = np.array(N_syn)

    return P_syn, N_syn


