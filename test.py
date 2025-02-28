import igl
import numpy as np
import open3d as o3d

# 交互式选顶点
def select_vertices(V, F):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)
    ################## 顶点选择
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.run()
    selected_points = vis.get_picked_points()
    vis.destroy_window()
    ################## 顶点索引
    indices = np.array([point.index for point in selected_points], dtype=np.int32)
    return np.array(indices, dtype=np.int32)

with np.load("/home/chli/chLi/Dataset/arap_test/delta.npz") as data:
    delta = data["delta"]

# V1, F1 = igl.read_triangle_mesh("/mnt/d/sdc/liutong/codes/Data/generateCodes/result.obj")
V1, F1 = igl.read_triangle_mesh("/home/chli/chLi/Dataset/arap_test/result.obj")

V2 = V1 + delta

target_mesh = o3d.geometry.TriangleMesh()
target_mesh.vertices = o3d.utility.Vector3dVector(V2)
target_mesh.triangles = o3d.utility.Vector3iVector(F1)

target_mesh.translate([1, 0, 0])

print('start select bnd')
# bnd = np.arange(V1.shape[0]) # 所有点都作为约束点
bnd = select_vertices(V1, F1) # 交互选择约束点
bc = V2[bnd] # 选择点的期望位置

arap = igl.ARAP(V1, F1, 3, bnd)

U = arap.solve(bc, V1)

print('U:')
print('====')
print(U)
print('====')

result = o3d.geometry.TriangleMesh()
result.vertices = o3d.utility.Vector3dVector(U)
result.triangles = o3d.utility.Vector3iVector(F1)
o3d.visualization.draw_geometries([target_mesh, result])
