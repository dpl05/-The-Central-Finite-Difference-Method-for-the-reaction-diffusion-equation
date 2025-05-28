import numpy as np
import matplotlib.pyplot as plt

# 空间和时间离散
Nx, Nt = 30, 30
L, T = 1.0, 1.0
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)
dx = x[1] - x[0]
dt = t[1] - t[0]

# 参数和函数定义 
def g(t):  # 时间部分
    #return np.sin(np.pi * t) + 1.1  # 始终正值，确保正则性
    return np.ones_like(t)  # 始终为 1，确保正则性

def sample_f_x(seed=None):  # 随机采样 f(x)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=Nx)

def sample_u0_x(seed=None):  # 随机采样初始条件
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=Nx)

def solve_reaction_diffusion(fx, u0, g, Nx=30, Nt=30):
    u = np.zeros((Nt, Nx))
    u[0, :] = u0

    # Neumann BC: Dirichlet u(0)=u(Nx-1)=0
    for n in range(0, Nt - 1):
        for i in range(1, Nx - 1):
            u[n+1, i] = (
                u[n, i]
                + dt / dx**2 * (u[n, i-1] - 2 * u[n, i] + u[n, i+1])
                + dt * fx[i] * g(t[n])
            )
    return u

# 生成一个样本
fx = sample_f_x(seed=1)
u0 = sample_u0_x(seed=2)
u = solve_reaction_diffusion(fx, u0, g)

# 可视化
plt.figure(figsize=(6, 4))
plt.imshow(u, extent=[0,1,0,1], origin='lower', aspect='auto', cmap='hot')
plt.colorbar(label='u(x,t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Solution u(x,t) of Reaction-Diffusion Equation')
plt.show()




# 获取用于训练的边界 + 初末态数据
boundary_data = np.hstack([u[:, 0:1], u[:, -1:]])  # u(0,t) 与 u(1,t)
initial_data = u[0, :]   # u(x,0)
final_data = u[-1, :]    # u(x,1)

## 保存用于监督训练的标签 f(x)
# np.savez('sample_data.npz',
#          u_boundary=boundary_data,
#          u0=initial_data,
#          uT=final_data,
#          fx=fx)

# import numpy as np
# import matplotlib.pyplot as plt

# # 网格参数
# Nx = 30
# Nt = 30
# x_start, x_end = 0, 1
# t_start, t_end = 0, 1

# # 网格划分
# x = np.linspace(x_start, x_end, Nx)
# t = np.linspace(t_start, t_end, Nt)
# dx = x[1] - x[0]
# dt = t[1] - t[0]

# # 初始化解矩阵 u(x, t)
# u = np.zeros((Nt, Nx))

# # 定义 f(x)
# def f(x):
#     return 5*np.sin(np.pi/2 * x)  # 可自定义

# # 初始条件：u(x, 0) = 0（可更改）
# u[0, :] = 0

# # 边界条件：u(0, t) = u(1, t) = 0（可更改）
# # 时间步推进
# for n in range(0, Nt - 1):
#     for i in range(1, Nx - 1):
#         u[n+1, i] = u[n, i] - dt * (
#             (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx**2 - f(x[i])
#         )
#     # 边界条件
#     u[n+1, 0] = 0
#     u[n+1, -1] = 0

# # 给出出初始分布
# u0 = np.zeros(Nx)

# # 给出终止分布
# uT = np.zeros(Nx)

# # 画二维热图（正向 t 轴）
# plt.figure(figsize=(6, 5))
# plt.imshow(u, extent=[x_start, x_end, t_start, t_end], 
#            aspect='auto', cmap='hot', origin='lower')
# plt.colorbar(label='u(x,t)')
# plt.xlabel('x')
# plt.ylabel('t')
# plt.title('Heatmap of u(x,t)')
# plt.tight_layout()
# plt.show()
