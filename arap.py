import igl
import trimesh
import numpy as np
import open3d as o3d
from tqdm import trange
from scipy.sparse import lil_matrix

def compute_laplacian_matrix(vertices, faces):
    """
    计算顶点的拉普拉斯矩阵（基于参数化的面积加权）
    
    参数:
        vertices (np.ndarray): 顶点坐标 (n x 3)
        faces (np.ndarray): 三角面片索引 (m x 3)，存储顶点索引（从0开始）
    
    返回:
        csr_matrix: 对称化的拉普拉斯矩阵
    """
    n = vertices.shape[0]
    L = lil_matrix((n, n))  # 使用LIL格式初始化

    for face in faces:
        v0_idx, v1_idx, v2_idx = face  # 确保face存储的是顶点索引（整数）
        v0 = vertices[v0_idx]
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]

        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(normal)
        if area == 0:
            continue  # 跳过退化面

        w = area / 3.0  # 每个顶点的权重

        # 更新拉普拉斯矩阵
        # 注意：LIL矩阵支持直接通过索引赋值，无需使用+=
        L[v0_idx, v0_idx] += w
        L[v0_idx, v1_idx] -= w
        L[v0_idx, v2_idx] -= w
        
        L[v1_idx, v0_idx] -= w
        L[v1_idx, v1_idx] += w
        L[v1_idx, v2_idx] -= w
        
        L[v2_idx, v0_idx] -= w
        L[v2_idx, v1_idx] -= w
        L[v2_idx, v2_idx] += w
    
    # 对称化矩阵并转换为CSR格式
    L = (L + L.T) / 2
    return L.tocsr()

def arap_loss(vertices, faces, u, lambda_rigid=1.0):
    """ARAP能量函数计算"""
    n = vertices.shape[0]
    L = compute_laplacian_matrix(vertices, faces)
    
    # 计算刚性项
    R = u - vertices  # 当前位移
    rigid_loss = 0.5 * lambda_rigid * R.T @ L @ R
    
    # 计算平滑项（可选）
    # smooth_loss = 0.5 * alpha * R.T @ R  # alpha为平滑权重
    
    return rigid_loss  # + smooth_loss

def arap_optimization(vertices, faces, u_init, lambda_rigid=1.0, max_iter=100, tol=1e-6):
    """ARAP形变优化"""
    u = u_init.copy()
    for _ in trange(max_iter):
        # 计算ARAP损失及梯度
        grad = compute_arap_gradient(vertices, faces, u, lambda_rigid)
        
        # 更新位移场
        u -= 0.1 * grad  # 学习率设为0.1
        
        # 检查收敛
        if np.linalg.norm(grad) < tol:
            break
    
    return u

def compute_arap_gradient(vertices, faces, u, lambda_rigid):
    """计算ARAP梯度（简化版）"""
    n = vertices.shape[0]
    L = compute_laplacian_matrix(vertices, faces)
    R = u - vertices
    grad = lambda_rigid * L @ R
    return grad

with np.load("/home/chli/chLi/Dataset/arap_test/delta.npz") as data:
    delta = data["delta"]

# 读取三角网格（需自行实现或使用第三方库）
o3d_mesh = o3d.io.read_triangle_mesh("/home/chli/chLi/Dataset/arap_test/result.obj")
tmesh = trimesh.load("/home/chli/chLi/Dataset/arap_test/result.obj")
V1, F1 = igl.read_triangle_mesh("/home/chli/chLi/Dataset/arap_test/result.obj")

o3d_V1 = np.asarray(o3d_mesh.vertices)
o3d_F1 = np.asarray(o3d_mesh.triangles)

tV1 = tmesh.vertices
tF1 = tmesh.faces

print(V1.shape, o3d_V1.shape, tV1.shape)
print(F1.shape, o3d_F1.shape, tF1.shape)
print(delta.shape)

V2 = V1 + delta

source_mesh = o3d.geometry.TriangleMesh()
source_mesh.vertices = o3d.utility.Vector3dVector(V1)
source_mesh.triangles = o3d.utility.Vector3iVector(F1)

source_mesh.translate([-1, 0, 0])

target_mesh = o3d.geometry.TriangleMesh()
target_mesh.vertices = o3d.utility.Vector3dVector(V2)
target_mesh.triangles = o3d.utility.Vector3iVector(F1)

target_mesh.translate([-0.5, 0, 0])

# 执行ARAP优化
u_final = arap_optimization(V1, F1, V2, lambda_rigid=1e5)

print('U:')
print('====')
print(u_final)
print('====')

# 更新顶点位置
U2 = V1 + u_final

arap_mesh = o3d.geometry.TriangleMesh()
arap_mesh.vertices = o3d.utility.Vector3dVector(U2)
arap_mesh.triangles = o3d.utility.Vector3iVector(F1)

o3d.visualization.draw_geometries([source_mesh, target_mesh, arap_mesh])
