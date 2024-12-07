import numpy as np
import sys
import pathlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

file_path = pathlib.Path(__file__)

root_dir = file_path.parent.parent.parent
sys.path.append(str(root_dir))
from common.gif_creator import *
from common.plot_util import *
from tests.optimization.optimize_method_test import *

# 计算函数在x点的值
def function(x) -> float:
    return x[0] ** 2 + 6 * x[1] ** 2


# 计算函数在x点的梯度值
def function_gradient(x):
    return np.array([x[0] * 2, x[1] * 12])


# 输入：x:当前的搜索点; d:梯度的负方向; 实际输入：x_i[i], -gradient
# 输出：满足armijo条件的步长alpha
# 功能：输入初始步长alpha为1.0，如果该步长不满足armijo条件，怎每次将步长变为原来的0.9倍，直到找到满足条件的步长并返回。
#      (注：取方向与梯度点积，相当于取梯度的平方的负数。即为了提出梯度正负的影响，保证armijo条件的后半部分是小于0的。)
def armijo(x, d) -> float:
    c1 = 1e-3
    gamma = 0.9
    alpha = 1.0 # 初始步长

    while function(x + alpha * d) > function(x) + c1 * alpha * np.dot(
        function_gradient(x).T, d
    ):
        alpha = gamma * alpha
    return alpha


# 输入：x:当前的搜索点; d:梯度的负方向; 实际输入：x_i[i], -gradient
# 输出：满足goldstein条件的步长alpha
# 功能：c1与c2给定后不再改变。改定初始步长1.0(alpha)后，如果步长太大或者太小，缩放步长使f(x+alpha*d)落在l1~l2之间
def goldstein(x, d):

    a = 0       # a,b是在步长太大或者太小时，缩放步长时使用的
    b = np.inf  # 同上
    alpha = 1   # 步长
    c1 = 0.1    # 可接受系数
    c2 = 1 - c1 # (c1取0.0~0.5，是为了保证c2>c1,且两者都属于0.0~1.0)
    beta = 2    # 试探点系数(如果步长太小时，使用的放大系数)

    # 当更新步长时，两次步长插值的绝对值<1e-5时，会强制结束
    while np.fabs(a - b) > 1e-5:
        if function(x + alpha * d) <= function(x) + c1 * alpha * np.dot(
            function_gradient(x).T, d
        ):
            if function(x + alpha * d) >= function(x) + c2 * alpha * np.dot(
                function_gradient(x).T, d
            ):
                # 3. f(x+alpha*d)满足<l1, >l2，即落在两条直线中间。结束循环
                break
            else:
                # 2. f(x+alpha*d)满足<l1，<l2说明步长太小需要放大。若步长原来用0.5缩小过，则让步长靠近b;否则每次将步长放大2倍
                a = alpha
                # alpha = (a + b) / 2
                if b < np.inf:
                    alpha = (a + b) / 2
                else:
                    alpha = beta * alpha
        else:
            # 1. f(x+alpha*d) > l1,也肯定>l2(即连Armijo都不满足),说明步长太大需要缩小。调整步长alpha，每次缩短为原来的1/2
            b = alpha
            alpha = (a + b) / 2

    return alpha


# 输入：x:当前的搜索点; d:梯度的负方向; 实际输入：x_i[i], -gradient
# 输出：满足Wolfe条件的步长alpha
# 功能：c1,c2取经验值且不再变。改定初始步长1.0(alpha)后，如果步长太大或者太小，缩放步长以满足Wolfe条件
def wolfe(x, d):

    c1 = 0.3
    c2 = 0.9
    alpha = 1   # 步长
    a = 0       # a,b是在步长太大或者太小时，缩放步长时使用的
    b = np.inf  # 同上

    # 当*时，会强制结束
    while a < b:
        if function(x + alpha * d) <= function(x) + c1 * alpha * np.dot(
            function_gradient(x).T, d
        ):
            if np.dot(function_gradient(x + alpha * d).T, d) >= c2 * alpha * np.dot(
                function_gradient(x).T, d
            ):
                # 3. 满足Wolfe条件，返回
                break
            else:
                # 2. 如果满足Armijo条件，但不满足Wolfe准则，则放大alpha逐渐靠近b
                a = alpha
                alpha = (a + b) / 2
        else:
            # 1. f(x+alpha*d) > l(即不满足Armijo条件),说明步长太大需要缩小。调整步长alpha，每次缩短为原来的1/2
            b = alpha
            alpha = (a + b) / 2

    return alpha


# 输入： x0:初始值; line_search:线搜索方法函数; iterations:最大迭代次数;
# 输出： solution:; x_i:;
def gradient_descent_optimize(x0, line_search, iterations=1000):
    # 记录每次梯度下降法得到的x值
    x_i = [x0]
    for i in range(iterations):
        gradient = function_gradient(x_i[i])
        # 通过线搜索方法得到步长alpha
        alpha = line_search(x_i[i], -gradient)
        x_i.append(x_i[i] - alpha * gradient)

        # 梯度下降法结束的判断(梯度约等于0时，找到最优解)
        if np.linalg.norm(gradient) < 10e-5:
            solution = x_i[i + 1]
            print(f"\nConvergence Achieved ({i+1} iterations): Solution = {solution}")
            break
        else:
            solution = None

        print(f"Step {i+1}:{x_i[i+1]}")

    return solution, x_i


def line_search_test():
    # 定义初始点
    x0 = np.array([-5, 8])

    # 分别用3中不同的线搜索方法，得到优化结果
    solution, armijo_x_i = gradient_descent_optimize(copy.deepcopy(x0), armijo)
    solution, goldstein_x_i = gradient_descent_optimize(copy.deepcopy(x0), goldstein)
    solution, wolfe_x_i = gradient_descent_optimize(copy.deepcopy(x0), wolfe)

    # 画图
    fig, ax = plt.subplots()
    plot_x_0 = [x_i[0] for x_i in armijo_x_i]
    plot_x_1 = [x_i[1] for x_i in armijo_x_i]
    plt.plot(plot_x_0, plot_x_1, "r*-", label="armijo")

    plot_x_0 = [x_i[0] for x_i in goldstein_x_i]
    plot_x_1 = [x_i[1] for x_i in goldstein_x_i]
    plt.plot(plot_x_0, plot_x_1, "g*-", label="goldstein")

    plot_x_0 = [x_i[0] for x_i in wolfe_x_i]
    plot_x_1 = [x_i[1] for x_i in wolfe_x_i]
    plt.plot(plot_x_0, plot_x_1, "y*-", label="wolfe")
    plt.plot(x0[0], x0[1], "ko", label="x0")

    for i in range(11):
        ax.add_patch(Circle((0, 0), i, facecolor="k", alpha=0.3))

    plt.title("Line Search Test")
    ax.axis("equal")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    line_search_test()
