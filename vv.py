import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 字体配置 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# ================= 2. 准备数据 =================
n = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
h = 1 / n  # 步长 (x轴数据)
pi_true = np.pi

# 原始误差 (蓝色曲线, 斜率≈2.00)
pi_approx_1 = n * np.sin(np.pi / n)
error_1 = np.abs(pi_true - pi_approx_1)

# 模拟外推误差 (红色曲线, 斜率≈9.76)
error_2 = 0.5 * (h ** 9.76)

# ================= 3. 绘制双对数折线图 =================
plt.figure(figsize=(12, 8))

# 绘制曲线，图例中加入斜率值
plt.loglog(h, error_1, 'o-', color='blue', label=r'原始误差 ($e_n$), slope \(\approx\) 2.00', linewidth=2)
plt.loglog(h, error_2, 's--', color='red', label=r'外推误差 (模拟), slope \(\approx\) 9.76', linewidth=2)

# 反转 x 轴
plt.gca().invert_xaxis()

# 设置坐标轴标签
plt.xlabel(r'步长 $h = 1/n$', fontsize=14)
plt.ylabel(r'绝对误差 $|\pi - \pi_n|$', fontsize=14)
plt.title('有限元法求圆周率收敛速率对比', fontsize=16)
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

# ================= 4. 添加原图中的数字标注 (细节保留) =================
# 蓝色曲线上的小标签 (1.46, 1.87, 1.97, 1.99, 2.00, 2.00, 2.00)
upper_values = ["1.46", "1.87", "1.97", "1.99", "2.00", "2.00", "2.00"]
for i in range(1, 8):
    x_mid = np.sqrt(h[i] * h[i+1])
    y_mid = np.sqrt(error_1[i] * error_1[i+1])
    plt.text(x_mid, y_mid * 1.8, upper_values[i-1], fontsize=11, color='black', ha='center')

# 红色曲线上的小标签 (5.31, 7.50, 9.76)
lower_values = ["5.31", "7.50", "9.76"]
for i, val in enumerate(lower_values):
    idx = i + 3
    if idx < len(h) - 1:
        x_mid = np.sqrt(h[idx] * h[idx+1])
        y_mid = np.sqrt(error_2[idx] * error_2[idx+1])
        plt.text(x_mid, y_mid * 0.5, val, fontsize=11, color='black', ha='center')

# ================= 5. 添加用户要求的箭头标注 (slope: 2.00 & 9.76) =================
# --- 标注蓝色曲线 “slope：2.00” 并带箭头 ---
arrow_idx_blue = 6  # 选择 n=64 对应的点 (h=0.015625)
x_blue = h[arrow_idx_blue]
y_blue = error_1[arrow_idx_blue]

plt.annotate(
    'slope：2.00',
    xy=(x_blue, y_blue),             # 箭头指向的坐标 (蓝色曲线上)
    xytext=(x_blue * 0.6, y_blue * 3.5), # 文字显示的坐标 (右上方)
    fontsize=14,
    fontweight='bold',
    color='darkblue',
    arrowprops=dict(
        arrowstyle='->', 
        color='black', 
        lw=2,
        connectionstyle="arc3,rad=-0.2"  # 稍微带点弧度
    )
)

# --- 标注红色曲线 “slope：9.76” 并带箭头 ---
arrow_idx_red = 5  # 选择 n=32 对应的点 (h=0.03125)
x_red = h[arrow_idx_red]
y_red = error_2[arrow_idx_red]

plt.annotate(
    'slope：9.76',
    xy=(x_red, y_red),             # 箭头指向的坐标 (红色曲线上)
    xytext=(x_red * 0.7, y_red * 1.2),  # 文字显示的坐标 (右下方)
    fontsize=14,
    fontweight='bold',
    color='darkred',
    arrowprops=dict(
        arrowstyle='->', 
        color='black', 
        lw=2,
        connectionstyle="arc3,rad=-0.2"
    )
)

# ================= 6. 显示图表 =================
plt.tight_layout()
plt.show()
