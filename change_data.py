import igl
import numpy as np
import open3d as o3d

data_folder_path = '/home/chli/chLi/Dataset/arap_test/'

with np.load(data_folder_path + "delta.npz") as data:
    delta = data["delta"]

V1, F1 = igl.read_triangle_mesh(data_folder_path + "result.obj")

V2 = V1 + delta

target_mesh = o3d.geometry.TriangleMesh()
target_mesh.vertices = o3d.utility.Vector3dVector(V2)
target_mesh.triangles = o3d.utility.Vector3iVector(F1)

with open(data_folder_path + 'result_xyz.txt', 'w') as f:
    for point in V1:
        f.write(str(point[0]) + ',' + str(point[1]) + ',' + str(point[2]) + '\n')

with open(data_folder_path + 'delta_xyz.txt', 'w') as f:
    for point in delta:
        f.write(str(point[0]) + ',' + str(point[1]) + ',' + str(point[2]) + '\n')

o3d.io.write_triangle_mesh(data_folder_path + 'target_mesh.obj', target_mesh, write_ascii=True)

V_target, F_target = igl.read_triangle_mesh(data_folder_path + "target_mesh.obj")

assert V_target.shape[0] == V1.shape[0]
