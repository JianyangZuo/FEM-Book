import numpy as np

def truss3d_element_stiffness(x1, x2, E, A):
    """
    计算三维杆单元 / 空间桁架单元的长度、方向余弦和单元刚度矩阵。

    Parameters
    ----------
    x1 : array_like
        节点 1 坐标，例如 [x1, y1, z1]
    x2 : array_like
        节点 2 坐标，例如 [x2, y2, z2]
    E : float
        弹性模量，单位 Pa
    A : float
        截面面积，单位 m^2

    Returns
    -------
    L : float
        单元长度
    direction_cosines : ndarray
        方向余弦 [cx, cy, cz]
    Ke : ndarray
        三维杆单元在全局坐标系下的 6 x 6 刚度矩阵
    """

    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    if x1.shape != (3,) or x2.shape != (3,):
        raise ValueError("节点坐标 x1 和 x2 必须是长度为 3 的数组，例如 [x, y, z]。")

    if E <= 0:
        raise ValueError("弹性模量 E 必须大于 0。")

    if A <= 0:
        raise ValueError("截面面积 A 必须大于 0。")

    # 两节点坐标差
    dx = x2 - x1

    # 单元长度
    L = np.linalg.norm(dx)

    # 退化单元检查
    if L <= 1.0e-12:
        raise ValueError("错误：两个节点重合，单元长度为 0，不能计算刚度矩阵。")

    # 方向余弦
    cx, cy, cz = dx / L
    direction_cosines = np.array([cx, cy, cz])

    # 构造方向余弦向量
    c = np.array([cx, cy, cz])

    # 3 x 3 子矩阵
    k3 = np.outer(c, c)

    # 三维杆单元在全局坐标系下的 6 x 6 刚度矩阵
    Ke = (E * A / L) * np.block([
        [ k3, -k3],
        [-k3,  k3]
    ])

    return L, direction_cosines, Ke

def truss3d_element_stress(x1, x2, E, A, de):
    """
    根据单元节点位移计算三维杆单元的轴向应变、应力和轴力。

    Parameters
    ----------
    x1 : array_like
        节点 1 坐标 [x1, y1, z1]
    x2 : array_like
        节点 2 坐标 [x2, y2, z2]
    E : float
        弹性模量，单位 Pa
    A : float
        截面面积，单位 m^2
    de : array_like
        单元节点位移列阵 [u1, v1, w1, u2, v2, w2]

    Returns
    -------
    epsilon : float
        单元轴向应变
    sigma : float
        单元轴向应力，单位 Pa
    N : float
        单元轴力，单位 N
    """

    de = np.asarray(de, dtype=float)

    if de.shape != (6,):
        raise ValueError("单元节点位移 de 必须是长度为 6 的数组：[u1, v1, w1, u2, v2, w2]。")

    L, direction_cosines, _ = truss3d_element_stiffness(x1, x2, E, A)

    cx, cy, cz = direction_cosines

    # 应变-位移矩阵 B
    # epsilon = B * de
    B = np.array([-cx, -cy, -cz, cx, cy, cz]) / L

    # 轴向应变
    epsilon = B @ de

    # 轴向应力
    sigma = E * epsilon

    # 轴力
    N = sigma * A

    return epsilon, sigma, N

def element_internal_force(x1, x2, E, A, de):
    """
    计算单元节点内力列阵：

        Fe = Ke * de

    Parameters
    ----------
    x1, x2 : array_like
        两个节点坐标
    E : float
        弹性模量
    A : float
        截面面积
    de : array_like
        单元节点位移列阵

    Returns
    -------
    Fe : ndarray
        单元节点内力列阵
    """

    de = np.asarray(de, dtype=float)

    if de.shape != (6,):
        raise ValueError("单元节点位移 de 必须是长度为 6 的数组。")

    _, _, Ke = truss3d_element_stiffness(x1, x2, E, A)

    Fe = Ke @ de

    return Fe

def check_stiffness_matrix_properties(Ke, tol=1.0e-8):
    """
    检查刚度矩阵性质：
    1. 对称性
    2. 特征值
    3. 半正定性
    4. 奇异性
    """

    # 检查对称性
    is_symmetric = np.allclose(Ke, Ke.T, atol=tol)

    # 对称矩阵特征值
    eigenvalues = np.linalg.eigvalsh(Ke)

    # 检查半正定性
    is_positive_semidefinite = np.all(eigenvalues >= -tol)

    # 矩阵秩
    rank = np.linalg.matrix_rank(Ke, tol=tol)

    # 检查奇异性
    is_singular = rank < Ke.shape[0]

    result = {
        "is_symmetric": is_symmetric,
        "eigenvalues": eigenvalues,
        "is_positive_semidefinite": is_positive_semidefinite,
        "rank": rank,
        "is_singular": is_singular
    }

    return result

def check_rigid_body_translation(x1, x2, E, A):
    """
    检查刚体平移位移是否产生内力。

    如果两个节点发生相同的平移位移，则杆件没有伸长，
    因此应变、应力、轴力和节点内力都应为 0。
    """

    de_rigid = np.array([
        0.001, -0.002, 0.003,
        0.001, -0.002, 0.003
    ])

    Fe = element_internal_force(x1, x2, E, A, de_rigid)

    epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de_rigid)

    return de_rigid, Fe, epsilon, sigma, N

def verify_column_physical_meaning(x1, x2, E, A, j):
    """
    任务 4：验证刚度矩阵第 j 列的物理意义。

    任意选择一个自由度 j，令该自由度位移为 1，
    其他自由度位移为 0，计算：

        Fe = Ke * de

    此时得到的 Fe 应当等于刚度矩阵 Ke 的第 j 列。

    Parameters
    ----------
    x1, x2 : array_like
        两个节点坐标
    E : float
        弹性模量
    A : float
        截面面积
    j : int
        自由度编号，取值 0 到 5

        Python 中编号从 0 开始：
        j = 0 表示 u1
        j = 1 表示 v1
        j = 2 表示 w1
        j = 3 表示 u2
        j = 4 表示 v2
        j = 5 表示 w2

    Returns
    -------
    de_unit : ndarray
        单位位移向量
    Fe : ndarray
        Fe = Ke * de_unit
    column_j : ndarray
        Ke 的第 j 列
    is_equal : bool
        Fe 是否等于 Ke 的第 j 列
    """

    if j < 0 or j > 5:
        raise ValueError("自由度编号 j 必须在 0 到 5 之间。")

    _, _, Ke = truss3d_element_stiffness(x1, x2, E, A)

    # 构造单位位移列阵
    # 第 j 个自由度位移为 1，其他自由度位移为 0
    de_unit = np.zeros(6)
    de_unit[j] = 1.0

    # 计算节点力列阵
    Fe = Ke @ de_unit

    # 提取刚度矩阵第 j 列
    column_j = Ke[:, j]

    # 判断二者是否相等
    is_equal = np.allclose(Fe, column_j)

    return de_unit, Fe, column_j, is_equal

def print_matrix(name, matrix):
    """
    格式化输出矩阵，便于阅读。
    """

    print(f"{name} =")
    with np.printoptions(precision=6, suppress=False):
        print(matrix)
    print()

def run_example_1():
    """
    算例 1：沿 x 轴的一维杆单元
    """

    print("=" * 80)
    print("算例 1：沿 x 轴的一维杆单元")
    print("=" * 80)

    x1 = [0.0, 0.0, 0.0]
    x2 = [2.0, 0.0, 0.0]

    E = 200.0e9
    A = 1.0e-4

    de = [0.0, 0.0, 0.0,
          1.0e-3, 0.0, 0.0]

    L, direction_cosines, Ke = truss3d_element_stiffness(x1, x2, E, A)

    epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)

    Fe = element_internal_force(x1, x2, E, A, de)

    print(f"节点 1 坐标 x1 = {x1}")
    print(f"节点 2 坐标 x2 = {x2}")
    print(f"E = {int(E/1e9)} GPa")
    print(f"A = {A:.6e} m^2")
    print(f"de = {de}^T m")
    print()

    print(f"单元长度 L = {L:.6f} m")
    print(f"方向余弦 [cx, cy, cz] = {direction_cosines}")
    print()

    print_matrix("单元刚度矩阵 Ke", Ke)

    print(f"轴向应变 epsilon = {epsilon:.6e}")
    print(f"轴向应力 sigma = {sigma:.6e} Pa = {sigma / 1.0e6:.6f} MPa")
    print(f"轴力 N = {N:.6e} N")
    print()

    print_matrix("单元节点内力 Fe = Ke * de", Fe)

    properties = check_stiffness_matrix_properties(Ke)

    print("刚度矩阵性质检查：")
    print(f"Ke 是否对称: {properties['is_symmetric']}")
    print(f"Ke 的秩 rank = {properties['rank']}")
    print(f"Ke 是否奇异: {properties['is_singular']}")
    print(f"Ke 是否半正定: {properties['is_positive_semidefinite']}")
    print("Ke 的特征值:")
    print(properties["eigenvalues"])
    print()

def run_example_2():
    """
    算例 2：空间任意方向杆单元
    """

    print("=" * 80)
    print("算例 2：空间任意方向杆单元")
    print("=" * 80)

    x1 = [0.0, 0.0, 0.0]
    x2 = [1.0, 2.0, 2.0]

    E = 210.0e9
    A = 2.0e-4

    de = [0.0, 0.0, 0.0,
          1.0e-3, 2.0e-3, 2.0e-3]

    L, direction_cosines, Ke = truss3d_element_stiffness(x1, x2, E, A)

    epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)

    Fe = element_internal_force(x1, x2, E, A, de)

    print(f"节点 1 坐标 x1 = {x1}")
    print(f"节点 2 坐标 x2 = {x2}")
    print(f"E = {int(E/1e9)} Pa")
    print(f"A = {A:.6e} m^2")
    print(f"de = {de} m")
    print()

    print(f"单元长度 L = {L:.6f} m")
    print(f"方向余弦 [cx, cy, cz] = {direction_cosines}")
    print()

    print_matrix("单元刚度矩阵 Ke", Ke)

    print(f"轴向应变 epsilon = {epsilon:.6e}")
    print(f"轴向应力 sigma = {sigma:.6e} Pa = {sigma / 1.0e6:.6f} MPa")
    print(f"轴力 N = {N:.6e} N")
    print()

    print_matrix("单元节点内力 Fe = Ke * de", Fe)

    properties = check_stiffness_matrix_properties(Ke)

    print("刚度矩阵性质检查：")
    print(f"Ke 是否对称: {properties['is_symmetric']}")
    print(f"Ke 的秩 rank = {properties['rank']}")
    print(f"Ke 是否奇异: {properties['is_singular']}")
    print(f"Ke 是否半正定: {properties['is_positive_semidefinite']}")
    print("Ke 的特征值:")
    print(properties["eigenvalues"])
    print()

    print("奇异性说明：")
    print("单个自由三维杆单元只能抵抗沿杆轴方向的拉压变形，")
    print("不能抵抗横向运动和刚体运动。")
    print("因此存在非零位移使单元不产生内力，所以 Ke 的秩小于 6，是奇异矩阵。")
    print()

    de_rigid, Fe_rigid, eps_rigid, sig_rigid, N_rigid = check_rigid_body_translation(x1, x2, E, A)

    print("刚体平移位移检查：")
    print(f"刚体平移位移 de_rigid = {de_rigid}")
    print_matrix("刚体平移对应的节点内力 Fe_rigid", Fe_rigid)
    print(f"刚体平移对应的 epsilon = {eps_rigid:.6e}")
    print(f"刚体平移对应的 sigma = {sig_rigid:.6e} Pa")
    print(f"刚体平移对应的 N = {N_rigid:.6e} N")
    print("可以看到，刚体平移不会产生应变、应力、轴力和节点内力。")
    print()

def run_task_4_physical_meaning():
    """
    任务 4：刚度矩阵物理意义验证。

    任意选一个自由度 j，令该自由度位移为 1，
    其他自由度位移为 0，计算：

        Fe = Ke * de

    并验证 Fe 是否等于 Ke 的第 j 列。
    """

    print("=" * 80)
    print("任务 4：刚度矩阵物理意义验证")
    print("=" * 80)

    # 这里采用算例 2 的空间任意方向杆单元进行验证
    x1 = [0.0, 0.0, 0.0]
    x2 = [1.0, 2.0, 2.0]

    E = 210.0e9
    A = 2.0e-4

    L, direction_cosines, Ke = truss3d_element_stiffness(x1, x2, E, A)

    print(f"节点 1 坐标 x1 = {x1}")
    print(f"节点 2 坐标 x2 = {x2}")
    print(f"E = {E:.6e} Pa")
    print(f"A = {A:.6e} m^2")
    print(f"单元长度 L = {L:.6f} m")
    print(f"方向余弦 [cx, cy, cz] = {direction_cosines}")
    print()

    print_matrix("单元刚度矩阵 Ke", Ke)

    # 可任意选择一个自由度 j
    # Python 索引从 0 开始：
    # j = 0 表示第 1 个自由度 u1
    # j = 1 表示第 2 个自由度 v1
    # j = 2 表示第 3 个自由度 w1
    # j = 3 表示第 4 个自由度 u2
    # j = 4 表示第 5 个自由度 v2
    # j = 5 表示第 6 个自由度 w2
    j = 3

    de_unit, Fe, column_j, is_equal = verify_column_physical_meaning(x1, x2, E, A, j)

    print(f"选择第 {j + 1} 个自由度进行验证。")
    print("即令该自由度位移为 1，其他自由度位移为 0。")
    print()

    print_matrix("单位位移列阵 de", de_unit)

    print("根据单元刚度方程：")
    print("Fe = Ke * de")
    print()

    print_matrix("计算得到的节点力列阵 Fe", Fe)

    print_matrix(f"刚度矩阵 Ke 的第 {j + 1} 列", column_j)

    print(f"Fe 是否等于 Ke 的第 {j + 1} 列: {is_equal}")
    print()

    print("结论：")
    print("当单元第 j 个自由度发生单位位移，而其他自由度位移为 0 时，")
    print("由 Fe = Ke * de 计算得到的节点力列阵 Fe 正好等于刚度矩阵 Ke 的第 j 列。")
    print()
    print("因此，刚度矩阵元素 k_ij 的物理意义为：")
    print("当单元第 j 个自由度发生单位位移，而其他自由度固定时，")
    print("为了使单元保持平衡，需要在单元第 i 个自由度上施加的节点力。")
    print()

def test_degenerate_element():
    """
    测试退化单元：两个节点重合。
    """

    print("=" * 80)
    print("退化单元检查")
    print("=" * 80)

    x1 = [0.0, 0.0, 0.0]
    x2 = [0.0, 0.0, 0.0]

    E = 200.0e9
    A = 1.0e-4

    try:
        truss3d_element_stiffness(x1, x2, E, A)
    except ValueError as e:
        print(e)
    print()

def main():
    """
    主函数。
    """

    np.set_printoptions(linewidth=120)

    run_example_1()
    run_example_2()

    # 任务 4：刚度矩阵物理意义验证
    run_task_4_physical_meaning()

    # 退化单元检查
    test_degenerate_element()

if __name__ == "__main__":
    main()
