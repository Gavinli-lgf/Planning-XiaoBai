"""

Reeds Shepp path planner sample code

author Atsushi Sakai(@Atsushi_twi)
co-author Videh Patel(@videh25) : Added the missing RS paths

"""

import sys
import pathlib
import random

file_path = pathlib.Path(__file__)

root_dir = file_path.parent.parent.parent
sys.path.append(str(root_dir))
from math import sin, cos, atan2, sqrt, acos, pi, hypot
import numpy as np
from common.geometry import *
from common.plot_util import *
from common.common_util import *
from common.gif_creator import *

# 定义一个path类（记录一个完成path所包含的信息）
class Path:
    """
    Path data container
    """

    def __init__(self):
        # course segment length  (negative value is backward segment)
        self.lengths = []
        # course segment type char ("S": straight, "L": left, "R": right)
        self.ctypes = []
        self.L = 0.0  # Total lengths of the path
        self.x = []  # x positions
        self.y = []  # y positions
        self.yaw = []  # orientations [rad]
        self.directions = []  # directions (1:forward, -1:backward)


# 角度取模运算：输入rad，输出[-pi, pi)
def pi_2_pi(x):
    return angle_mod(x)


def mod2pi(x):
    # Be consistent with fmod in cplusplus here.
    v = np.mod(x, np.copysign(2.0 * math.pi, x))
    if v < -math.pi:
        v += 2.0 * math.pi
    else:
        if v > math.pi:
            v -= 2.0 * math.pi
    return v


# 输入：paths, travel_distances, steering_dirns, step_size
# 功能：用求得的信息生成path对象，并校验path长度与是否已经存在相同的结果。校验通过后就将path加入paths(否则paths不变)
# 输出：paths
def set_path(paths, lengths, ctypes, step_size):
    path = Path()
    path.ctypes = ctypes
    path.lengths = lengths
    path.L = sum(np.abs(lengths))

    # check same path exist
    for i_path in paths:
        type_is_same = i_path.ctypes == path.ctypes
        length_is_close = (sum(np.abs(i_path.lengths)) - path.L) <= step_size
        if type_is_same and length_is_close:
            return paths  # same path found, so do not insert path

    # check path is long enough
    if path.L <= step_size:
        return paths  # too short, so do not insert path

    paths.append(path)
    return paths

# 将cartisian坐标(x,y)转换未极坐标(r,theta)
def polar(x, y):
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    return r, theta


# 输入转换与归一后的目标点(x,y,dth)
def left_straight_left(x, y, phi):
    u, t = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 <= t <= math.pi:
        v = mod2pi(phi - t)
        if 0.0 <= v <= math.pi:
            return True, [t, u, v], ["L", "S", "L"]

    return False, [], []


def left_straight_right(x, y, phi):
    u1, t1 = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1**2
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = mod2pi(t1 + theta)
        v = mod2pi(t - phi)

        if (t >= 0.0) and (v >= 0.0):
            return True, [t, u, v], ["L", "S", "R"]

    return False, [], []


# 输入:转换与归一后的目标点(x,y,dth); 输出3部分:bool,[每一段长度(正负表示前进/后退)],[运动方式]
def left_x_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + math.pi / 2)
        u = mod2pi(math.pi - 2 * A)
        v = mod2pi(phi - t - u)
        return True, [t, -u, v], ["L", "R", "L"]

    return False, [], []


def left_x_right_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + math.pi / 2)
        u = mod2pi(math.pi - 2 * A)
        v = mod2pi(-phi + t + u)
        return True, [t, -u, -v], ["L", "R", "L"]

    return False, [], []


def left_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        u = math.acos(1 - u1**2 * 0.125)
        A = math.asin(2 * math.sin(u) / u1)
        t = mod2pi(-A + theta + math.pi / 2)
        v = mod2pi(t - u - phi)
        return True, [t, u, -v], ["L", "R", "L"]

    return False, [], []


def left_right_x_left_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    # Solutions refering to (2 < u1 <= 4) are considered sub-optimal in paper
    # Solutions do not exist for u1 > 4
    if u1 <= 2:
        A = math.acos((u1 + 2) * 0.25)
        t = mod2pi(theta + A + math.pi / 2)
        u = mod2pi(A)
        v = mod2pi(phi - t + 2 * u)
        if (t >= 0) and (u >= 0) and (v >= 0):
            return True, [t, u, -u, -v], ["L", "R", "L", "R"]

    return False, [], []


def left_x_right_left_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)
    u2 = (20 - u1**2) / 16

    if 0 <= u2 <= 1:
        u = math.acos(u2)
        A = math.asin(2 * math.sin(u) / u1)
        t = mod2pi(theta + A + math.pi / 2)
        v = mod2pi(t - phi)
        if (t >= 0) and (v >= 0):
            return True, [t, -u, -u, v], ["L", "R", "L", "R"]

    return False, [], []


def left_x_right90_straight_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = mod2pi(theta + A + math.pi / 2)
        v = mod2pi(t - phi + math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi / 2, -u, -v], ["L", "R", "S", "L"]

    return False, [], []


def left_straight_right90_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(math.sqrt(u1**2 - 4), 2)
        t = mod2pi(theta - A + math.pi / 2)
        v = mod2pi(t - phi - math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi / 2, -v], ["L", "S", "R", "L"]

    return False, [], []


def left_x_right90_straight_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = mod2pi(theta + math.pi / 2)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi / 2, -u, -v], ["L", "R", "S", "R"]

    return False, [], []


def left_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = mod2pi(theta)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi / 2, -v], ["L", "S", "L", "R"]

    return False, [], []


def left_x_right90_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 4.0:
        u = math.sqrt(u1**2 - 4) - 4
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = mod2pi(theta + A + math.pi / 2)
        v = mod2pi(t - phi)
        if (t >= 0) and (v >= 0):
            return (
                True,
                [t, -math.pi / 2, -u, -math.pi / 2, v],
                ["L", "R", "S", "L", "R"],
            )

    return False, [], []


def timeflip(travel_distances):
    return [-x for x in travel_distances]


def reflect(steering_directions):
    def switch_dir(dirn):
        if dirn == "L":
            return "R"
        elif dirn == "R":
            return "L"
        else:
            return "S"

    return [switch_dir(dirn) for dirn in steering_directions]


# 计算出所有的46条path(未离散)
def generate_path(q0, q1, max_curvature, step_size):
    # 将初始点(x,y,yaw)转换为(0,0,0);并将各个长度参数对转弯半径做归一化处理。
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    dth = q1[2] - q0[2]
    c = math.cos(q0[2])
    s = math.sin(q0[2])
    x = (c * dx + s * dy) * max_curvature
    y = (-s * dx + c * dy) * max_curvature
    step_size *= max_curvature

    # 定义最终12种运动方式的求解方法函数的入口path_functions(输入转换与归一后的目标点(x,y,dth))
    # 注：这12个函数的实现就是套公式了。
    # 输入: 转换与归一后的目标点(x,y,dth); 
    # 输出: bool,[每一段长度(正负表示前进/后退)],[运动方式];(3部分含义:flag,travel_distances,steering_dirns)
    paths = []
    path_functions = [
        left_straight_left,
        left_straight_right,  # CSC
        left_x_right_x_left,
        left_x_right_left,
        left_right_x_left,  # CCC
        left_right_x_left_right,
        left_x_right_left_x_right,  # CCCC
        left_x_right90_straight_left,
        left_x_right90_straight_right,  # CCSC
        left_straight_right90_x_left,
        left_straight_left90_x_right,  # CSCC
        left_x_right90_straight_left90_x_right,
    ]  # CCSCC

    # 根据“时间翻转、反射、后向变换”来求解
    for path_func in path_functions:
        flag, travel_distances, steering_dirns = path_func(x, y, dth)
        if flag:
            for distance in travel_distances:
                if (
                    0.1 * sum([abs(d) for d in travel_distances])
                    < abs(distance)
                    < step_size
                ):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = path_func(-x, y, -dth)
        if flag:
            for distance in travel_distances:
                if (
                    0.1 * sum([abs(d) for d in travel_distances])
                    < abs(distance)
                    < step_size
                ):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            travel_distances = timeflip(travel_distances)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = path_func(x, -y, -dth)
        if flag:
            for distance in travel_distances:
                if (
                    0.1 * sum([abs(d) for d in travel_distances])
                    < abs(distance)
                    < step_size
                ):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            steering_dirns = reflect(steering_dirns)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = path_func(-x, -y, dth)
        if flag:
            for distance in travel_distances:
                if (
                    0.1 * sum([abs(d) for d in travel_distances])
                    < abs(distance)
                    < step_size
                ):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            travel_distances = timeflip(travel_distances)
            steering_dirns = reflect(steering_dirns)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

    return paths


# 输入：各段长度lengths,转换为弧度后的step_size
# 输出：各段离散化后的list的list（即interpolate_dists_list中的对象也是list，每个list对应一段的离散）
def calc_interpolate_dists_list(lengths, step_size):
    interpolate_dists_list = []
    for length in lengths:
        d_dist = step_size if length >= 0.0 else -step_size
        interp_dists = np.arange(0.0, length, d_dist)
        interp_dists = np.append(interp_dists, length)
        interpolate_dists_list.append(interp_dists)

    print(f"interpolate_dists_list shape: {len(interpolate_dists_list)}")
    return interpolate_dists_list


# 在local coordinate下对path离散化
def generate_local_course(lengths, modes, max_curvature, step_size):
    interpolate_dists_list = calc_interpolate_dists_list(
        lengths, step_size * max_curvature
    )

    origin_x, origin_y, origin_yaw = 0.0, 0.0, 0.0

    xs, ys, yaws, directions = [], [], [], []
    for interp_dists, mode, length in zip(interpolate_dists_list, modes, lengths):

        for dist in interp_dists:
            x, y, yaw, direction = interpolate(
                dist, length, mode, max_curvature, origin_x, origin_y, origin_yaw
            )
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
            directions.append(direction)
        origin_x = xs[-1]
        origin_y = ys[-1]
        origin_yaw = yaws[-1]

    return xs, ys, yaws, directions


# 
def interpolate(dist, length, mode, max_curvature, origin_x, origin_y, origin_yaw):
    if mode == "S":
        x = origin_x + dist / max_curvature * math.cos(origin_yaw)
        y = origin_y + dist / max_curvature * math.sin(origin_yaw)
        yaw = origin_yaw
    else:  # curve
        ldx = math.sin(dist) / max_curvature
        ldy = 0.0
        yaw = None
        if mode == "L":  # left turn
            ldy = (1.0 - math.cos(dist)) / max_curvature
            yaw = origin_yaw + dist
        elif mode == "R":  # right turn
            ldy = (1.0 - math.cos(dist)) / -max_curvature
            yaw = origin_yaw - dist
        gdx = math.cos(-origin_yaw) * ldx + math.sin(-origin_yaw) * ldy
        gdy = -math.sin(-origin_yaw) * ldx + math.cos(-origin_yaw) * ldy
        x = origin_x + gdx
        y = origin_y + gdy

    return x, y, yaw, 1 if length > 0.0 else -1

# 计算出所有的46条path，并对每条path进行离散化
def calc_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size):
    q0 = [sx, sy, syaw]
    q1 = [gx, gy, gyaw]
    
    # 计算出所有的46条path(未离散)
    paths = generate_path(q0, q1, maxc, step_size)
    for path in paths:
        # 在local coordinate下对path离散化
        xs, ys, yaws, directions = generate_local_course(
            path.lengths, path.ctypes, maxc, step_size
        )

        # convert global coordinate
        # 将离散后的local信息，转换为global信息
        path.x = [
            math.cos(-q0[2]) * ix + math.sin(-q0[2]) * iy + q0[0]
            for (ix, iy) in zip(xs, ys)
        ]
        path.y = [
            -math.sin(-q0[2]) * ix + math.cos(-q0[2]) * iy + q0[1]
            for (ix, iy) in zip(xs, ys)
        ]
        path.yaw = [pi_2_pi(yaw + q0[2]) for yaw in yaws]
        path.directions = directions
        path.lengths = [length / maxc for length in path.lengths]
        path.L = path.L / maxc

    return paths

# 根据输入的已知信息，计算所有共46条path，从中选出最短的path，并输出具体的离散后信息
def reeds_shepp_path_planning(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=0.2):
    # 计算出所有的46条path，并返回
    paths = calc_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size)
    if not paths:
        return None, None, None, None, None  # could not generate any path

    # search minimum cost path
    # 找出46条path中的长度最小的path
    best_path_index = paths.index(min(paths, key=lambda p: abs(p.L)))
    b_path = paths[best_path_index]

    # 返回最优path的具体信息(离散后的(x,y,yaw),ctypes,length)
    return b_path.x, b_path.y, b_path.yaw, b_path.ctypes, b_path.lengths


def main():
    print("Reeds Shepp path planner sample start!!")

    for i in range(0, 5):
        # 输入：7个已知量start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature
        start_x = random.uniform(-5, 5)  # [m]
        start_y = random.uniform(-5, 5)  # [m]
        start_yaw = np.deg2rad(random.uniform(-45, 45))  # [rad]

        end_x = random.uniform(-15, 15)  # [m]
        end_y = random.uniform(-15, 15)  # [m]
        end_yaw = np.deg2rad(random.uniform(145, 225))  # [rad]

        curvature = random.uniform(0.1, 0.3)
        step_size = 0.05

        # 获取reeds shepp规划结果(按step_size离散化后的结果)
        xs, ys, yaws, modes, lengths = reeds_shepp_path_planning(
            start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature, step_size
        )
        # 画图
        x_min, x_max, y_min, y_max = get_axes_limits(xs, ys, yaws)

        if not xs:
            assert False, "No path"

        for i in range(0, len(xs), max(int(len(xs) / 40), 1)):
            plt.cla()
            plt.plot(xs, ys, label=str(modes) + " length:" + get_num_str(sum(lengths)))

            plt.title("Reeds Shepp Path")
            plt.legend()
            plt.axis("equal")
            axes = fig.gca()
            axes.set_xlim(x_min, x_max)
            axes.set_ylim(y_min, y_max)
            plot_car(xs[i], ys[i], yaws[i])
            gif_creator.savefig()
            plt.pause(PAUSE_TIME)

        plt.cla()
        plt.plot(xs, ys, label=str(modes) + " length:" + get_num_str(sum(lengths)))
        plt.title("Reeds Shepp Path")
        plt.legend()
        plt.axis("equal")
        axes = fig.gca()
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)
        plot_car(xs[-1], ys[-1], yaws[-1])
        gif_creator.savefig()
        plt.pause(0.3)

    gif_creator.create_gif()


if __name__ == "__main__":
    fig, ax = plt.subplots()
    gif_creator = GifCreator(file_path, fig, ax)
    main()
