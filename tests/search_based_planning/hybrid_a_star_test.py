import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import sys
import pathlib

file_path = pathlib.Path(__file__)
sys.path.append(str(file_path.parent.parent.parent))
sys.path.append(str(file_path.parent.parent))


import copy
from common.geometry import *
from common.plot_util import *
from common.gif_creator import *
from common.common_util import *
from curves import reeds_shepp_path_test as rs

show_animation = True
show_heuristic_animation = False

############################ 1. Car Info ######################################
WB = 3.0  # rear to front wheel (轴距)
W = 2.0  # width of car (车宽)
LF = 3.3  # distance from rear to vehicle front end (后轴中心到车前边沿:bash link 2 front)
LB = 1.0  # distance from rear to vehicle back end (后轴中心到车后边沿)
MAX_STEER = 0.6  # [rad] maximum steering angle (最大转向角)

# distance from rear to center of vehicle. (后轴中心到质心距离)
BUBBLE_DIST = (LF - LB) / 2.0  
# bubble radius. (机器人尺寸最大半径，即车辆对角线长度的一半)
BUBBLE_R = np.hypot((LF + LB) / 2.0, W / 2.0)  

# vehicle rectangle vertices(车辆顶点，在后轴中心为原点的车身坐标系下的坐标，左前开始，顺时针方向。)
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]

# 输入： x_list,y_list,yaw_list:待检测曲线上每个离散点的位姿; ox,oy:实际地图; kd_tree:障碍物kd-tree;
# 输出： False:有碰撞; True:无碰撞
# 功能： 将待检测曲线上每个点转到质心，再取BUBBLE_R范围内的obs点，分别转换到车身坐标系下判碰(因此很耗时，TODO(lgf):看使用聪哥的方法会好在哪里??)
def check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
    for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
        # 将点从后轴中心(i_x,i_y)转移到质心(cx,xy)
        cx = i_x + BUBBLE_DIST * cos(i_yaw)
        cy = i_y + BUBBLE_DIST * sin(i_yaw)

        # 查找与质心(cx,xy)距离在r内的所有被obs占据的点。
        ids = kd_tree.query_ball_point([cx, cy], BUBBLE_R)

        if not ids:
            continue

        # 将obs的笛卡尔坐标(ox,oy)转换到base link frame下判碰(False:有碰撞; True:无碰撞)
        if not rectangle_check(
            i_x, i_y, i_yaw, [ox[i] for i in ids], [oy[i] for i in ids]
        ):
            return False  # collision

    return True  # no collision


# 输入：(x,y,yaw):自车后轴中心; ox,oy:与自车质心(cx,xy)距离在BUBBLE_R内的所有点列表。
# 输出：False:有碰撞; True:无碰撞
# 功能：将obs的笛卡尔坐标(ox,oy)转换到base link frame下判碰
def rectangle_check(x, y, yaw, ox, oy):
    # transform obstacles to base link frame
    # 将cartisian坐标(ox,oy)，转换到以后轴中心为原点的车身坐标下(base link frame)
    rot = rot_mat_2d(yaw)
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]

        # 在车身坐标下判碰
        if not (rx > LF or rx < -LB or ry > W / 2.0 or ry < -W / 2.0):
            return False  # collision

    return True  # no collision


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """Plot arrow."""
    if not isinstance(x, float):
        for i_x, i_y, i_yaw in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(
            x,
            y,
            length * cos(yaw),
            length * sin(yaw),
            fc=fc,
            ec=ec,
            head_width=width,
            head_length=width,
            alpha=0.4,
        )
    plt.savefig(gif_creator.get_image_path())


def plot_car(x, y, yaw):
    car_color = "-k"
    c, s = cos(yaw), sin(yaw)
    rot = rot_mat_2d(-yaw)
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0] + x)
        car_outline_y.append(converted_xy[1] + y)

    arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw

    plt.plot(car_outline_x, car_outline_y, car_color)


def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi


# 输入：(x,y,yaw):待拓展的cartisian位姿; [distance,steer]:[拓展的行驶距离(正负表示方向),输入转角]; L:轴距;
# 输出：拓展后的cartisian位姿(x,y,yaw)
# 功能：计算从(x,y,yaw)开始，以steer转角移动distance距离后的位姿，并返回
def move(x, y, yaw, distance, steer, L=WB):
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw += pi_2_pi(distance * tan(steer) / L)  # distance/2

    return x, y, yaw


############################ 2. dynamic programming heuristic #############################
# Node的简化版SimpleNode,只记录栅格坐标(x,y),代价cost(默认0),父节点索引parent_index(默认-1).
class SimpleNode:

    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return (
            str(self.x)
            + ","
            + str(self.y)
            + ","
            + str(self.cost)
            + ","
            + str(self.parent_index)
        )


def calc_final_path(goal_node, closed_node_set, resolution):
    # generate final course
    rx, ry = [goal_node.x * resolution], [goal_node.y * resolution]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_node_set[parent_index]
        rx.append(n.x * resolution)
        ry.append(n.y * resolution)
        parent_index = n.parent_index

    return rx, ry


# 输入：目标节点的真实坐标(gx,gy);其它解释如下：
# 输出：closed_set：已访问过节点的map(格式:Node索引，SimpleNode)
# 功能：用Dijkstra计算从goal_node到其它所有节点，考虑障碍物不考虑动力学的单源最短路径，作为距离启发值h
def calc_distance_heuristic(gx, gy, ox, oy, resolution, rr):
    """
    gx: goal x position [m]
    gy: goal y position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    resolution: grid resolution [m]
    rr: robot radius[m]
    """

    goal_node = SimpleNode(round(gx / resolution), round(gy / resolution), 0.0, -1)
    # 计算障碍物在栅格地图下的坐标列表ox,oy
    ox = [iox / resolution for iox in ox]
    oy = [ioy / resolution for ioy in oy]

    # 生成机器人的配置空间(Configuration Space Obstacle)
    obstacle_map, min_x, min_y, max_x, max_y, x_w, y_w = calc_obstacle_map(
        ox, oy, resolution, rr
    )

    # 定义8个方向，以及每个方向上cost
    motion = get_motion_model()

    # open_set/closed_set(格式:Node索引，SimpleNode)与priority_queue(格式:代价,索引)；都是记录待访问点。
    open_set, closed_set = dict(), dict()
    open_set[calculate_index(goal_node, x_w, min_x, min_y)] = goal_node # 把目标节点加入open_set
    # 优先队列格式(cost, c_id)
    priority_queue = [(0, calculate_index(goal_node, x_w, min_x, min_y))]

    while True:
        # 优先队列为空时，结束循环。
        if not priority_queue:
            break
        cost, c_id = heapq.heappop(priority_queue) # 取代价最小的点
        if c_id in open_set: # 如果当前点未访问过，则先标记为访问后，再继续后续访问处理
            current = open_set[c_id]
            closed_set[c_id] = current
            open_set.pop(c_id)
        else:
            continue

        # show graph
        if show_heuristic_animation:  # pragma: no cover
            plt.plot(current.x * resolution, current.y * resolution, "xc")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            if len(closed_set.keys()) % 10 == 0:
                plt.pause(0.001)

        # Remove the item from the open set

        # expand search grid based on motion model(向8个方向搜索栅格的邻居节点)
        for i, _ in enumerate(motion):
            node = SimpleNode(
                current.x + motion[i][0],
                current.y + motion[i][1],
                current.cost + motion[i][2],
                c_id,
            )
            n_id = calculate_index(node, x_w, min_x, min_y)

            # 如果该节点node已访问过，则continue
            if n_id in closed_set:
                continue

            # 如果node与障碍物发生碰撞，则continue
            if not verify_node(node, obstacle_map, min_x, min_y, max_x, max_y):
                continue

            # 将node加入open list或者进行松弛操作
            if n_id not in open_set:
                open_set[n_id] = node  # Discover a new node
                heapq.heappush(
                    priority_queue,
                    (node.cost, calculate_index(node, x_w, min_x, min_y)),
                )
            else:
                if open_set[n_id].cost >= node.cost:
                    # This path is the best until now. record it!
                    open_set[n_id] = node
                    heapq.heappush(
                        priority_queue,
                        (node.cost, calculate_index(node, x_w, min_x, min_y)),
                    )

    return closed_set


# 输入：node:待检测节点；obstacle_map:机器人配置空间地图；min_x,min_y,max_x,max_y:配置空间边界范围；
# 输出：node与障碍物是否有相交，False:相交；True:不相交
# 功能：判断node与障碍物是否发生碰撞
def verify_node(node, obstacle_map, min_x, min_y, max_x, max_y):
    if node.x < min_x:
        return False
    elif node.y < min_y:
        return False
    elif node.x >= max_x:
        return False
    elif node.y >= max_y:
        return False

    if obstacle_map[node.x][node.y]:
        return False

    return True


# 输入：ox,oy:障碍物在栅格地图下的坐标列表; 栅格分辨率resolution; 自车尺寸最大半径vr.
# 输出：Configuration Space Obstacle: obstacle_map, min_x, min_y, max_x, max_y, x_width, y_width
# 功能：在栅格地图中，将障碍物按照机器人最大半径进行拓展，生成配置空间(Configuration Space Obstacle)。这样机器人就可以当做点模型来搜索。
def calc_obstacle_map(ox, oy, resolution, vr):
    min_x = round(min(ox))
    min_y = round(min(oy))
    max_x = round(max(ox))
    max_y = round(max(oy))

    x_width = round(max_x - min_x)
    y_width = round(max_y - min_y)

    # obstacle map generation
    obstacle_map = [[False for _ in range(y_width)] for _ in range(x_width)]
    for ix in range(x_width):
        x = ix + min_x
        for iy in range(y_width):
            y = iy + min_y
            #  print(x, y)
            for iox, ioy in zip(ox, oy):
                d = math.hypot(iox - x, ioy - y)
                if d <= vr / resolution:
                    obstacle_map[ix][iy] = True
                    break

    return obstacle_map, min_x, min_y, max_x, max_y, x_width, y_width


# goal_node, x_w, min_x, min_y
# 输入：SimpleNode(node);栅格地图x宽度(x_width);栅格地图x,y的最小序号:x_min,y_min.
# 输出：node在栅格地图中的序号
# 功能：总列数*每列的点数，生成node在栅格地图中的唯一序号
def calculate_index(node, x_width, x_min, y_min):
    return (node.y - y_min) * x_width + (node.x - x_min)


# 定义8个方向，以及每个方向上cost
def get_motion_model():
    # dx, dy, cost
    motion = [
        [1, 0, 1],
        [0, 1, 1],
        [-1, 0, 1],
        [0, -1, 1],
        [-1, -1, math.sqrt(2)],
        [-1, 1, math.sqrt(2)],
        [1, -1, math.sqrt(2)],
        [1, 1, math.sqrt(2)],
    ]

    return motion


############################ 3. hybrid a star  ###########################################
XY_GRID_RESOLUTION = 2.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
MOTION_RESOLUTION = 0.1  # [m] path interpolate resolution (path离散化的ds)
N_STEER = 20  # number of steer command (转角输入采样数量。为了防止丢掉0.0,默认再加上0.0，共21个)

SB_COST = 100.0  # switch back penalty cost (若path中相邻2段采样曲线的正+反-行驶方向改变，则施加100.0的惩罚)
BACK_COST = 5.0  # backward penalty cost (对rs曲线中倒车-部分长度的惩罚系数)
STEER_CHANGE_COST = 5.0  # steer angle change penalty cost (rs曲线中正+反-同向，但是左L右R变化的惩罚系数)
STEER_COST = 1.0  # steer angle change penalty cost (rs曲线中C段转向角steer的惩罚系数)
H_COST = 5.0  # Heuristic cost(工程上的trick)


# 定义hybrid A*的节点结构体：(1) (x_index,y_index,yaw_index):栅格索引; (2) directions:行驶方向(True:正向;False:倒车),
# 其它成员信息为该节点到goal_node的one shot的rs曲线信息：
#   (3) (x_list,y_list,yaw_list):该Node之后控制采样得到的cartisian状态曲线(one shot时记录满足要求的rs曲线)；
#   (4) directions:每个点的行驶方向； (5) steer:转角(实际置为0.0); 
#   (6) parent_index:当前节点的位姿(x_index,y_index,yaw_index)的唯一索引;
#   (7) cost:当前节点的cost + 最优rs曲线的cost.
class Node:
    def __init__(
        self,
        x_ind,
        y_ind,
        yaw_ind,
        direction,
        x_list,
        y_list,
        yaw_list,
        directions,
        steer=0.0,
        parent_index=None,
        cost=None,
    ):
        self.x_index = x_ind
        self.y_index = y_ind
        self.yaw_index = yaw_ind
        self.direction = direction
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.directions = directions
        self.steer = steer
        self.parent_index = parent_index
        self.cost = cost


# Path类，包含各个离散点的cartisian信息(x,y,yaw,direction)与整体的cost
class Path:
    def __init__(self, x_list, y_list, yaw_list, direction_list, cost):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.direction_list = direction_list
        self.cost = cost


# 输入：障碍物的坐标信息，xy方向的分辨率为2.0m，yaw的分辨率为15度
# 输出：栅格化后地图的范围
# 功能：按照给定的地图信息，x,y,yaw的分辨率将地图栅格化，同时也对栅格编号并输出栅格范围
class Config:
    def __init__(self, ox, oy, xy_resolution, yaw_resolution):
        min_x_m = min(ox)
        min_y_m = min(oy)
        max_x_m = max(ox)
        max_y_m = max(oy)

        ox.append(min_x_m)
        oy.append(min_y_m)
        ox.append(max_x_m)
        oy.append(max_y_m)

        self.min_x = round(min_x_m / xy_resolution)
        self.min_y = round(min_y_m / xy_resolution)
        self.max_x = round(max_x_m / xy_resolution)
        self.max_y = round(max_y_m / xy_resolution)

        self.x_w = round(self.max_x - self.min_x)
        self.y_w = round(self.max_y - self.min_y)

        self.min_yaw = round(-math.pi / yaw_resolution) - 1
        self.max_yaw = round(math.pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw - self.min_yaw)


# 功能：使用yield生成器返回“[转角,行驶方向]”采样值
def calc_motion_inputs():
    # 对转角输入采样。为了防止丢掉0.0,默认再加上0.0
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER, N_STEER), [0.0])):
        for d in [1, -1]: # 1:向前; -1:倒车;
            yield [steer, d]


# 输入：current:待拓展节点; config:障碍物配置空间; ox,oy:实际地图; kd_tree:障碍物kd-tree;
# 输出：控制空间采样生成的无碰撞node
# 功能：对控制空间采样，并使用yield生成器返回采样生成neighbor的无碰撞node
def get_neighbors(current, config, ox, oy, kd_tree):
    # 使用yield生成器返回“[转角,行驶方向]”采样值
    for steer, d in calc_motion_inputs():
        # 在current之后的控制空间中以[steer,d]采样一段新的无碰撞曲线。并整理相关信息生成node
        node = calc_next_node(current, steer, d, config, ox, oy, kd_tree)
        # 如果生成了node，且node在栅格地图中：yield生成器返回node
        if node and verify_index(node, config):
            yield node


# 输入：current:待拓展节点; steer:转角采样值; direction:行驶方向采样值; config:障碍物配置空间; ox,oy:实际地图; kd_tree:障碍物kd-tree;
# 输出：根据该段控制空间采样曲线的起点新建的Node
# 功能：从current后接的采样曲线开始，以控制量[steer,direction]采样长度为"XY_GRID_RESOLUTION * 1.5"的曲线，如果无碰撞，则生成新的Node并返回
def calc_next_node(current, steer, direction, config, ox, oy, kd_tree):
    # 从该Node后接的采样曲线开始，以控制量[steer,direction]采样长度为"XY_GRID_RESOLUTION * 1.5"的曲线
    x, y, yaw = current.x_list[-1], current.y_list[-1], current.yaw_list[-1]

    arc_l = XY_GRID_RESOLUTION * 1.5
    x_list, y_list, yaw_list = [], [], []
    for _ in np.arange(0, arc_l, MOTION_RESOLUTION):
        # 计算从(x,y,yaw)开始，以steer转角移动(MOTION_RESOLUTION * direction)距离后的位姿
        x, y, yaw = move(x, y, yaw, MOTION_RESOLUTION * direction, steer)
        x_list.append(x)
        y_list.append(y)
        yaw_list.append(yaw)

    # 检测:自车沿该控制空间采样曲线行走时，会不会与obs产生碰撞。(False:有碰撞; True:无碰撞)
    if not check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
        return None

    d = direction == 1
    x_ind = round(x / XY_GRID_RESOLUTION)
    y_ind = round(y / XY_GRID_RESOLUTION)
    yaw_ind = round(yaw / YAW_GRID_RESOLUTION)

    # 计算这段控制空间采样曲线的代价(当前点cost + "形式方向变化，转角大小，转角变化量" + 曲线长度)
    added_cost = 0.0

    if d != current.direction:
        added_cost += SB_COST

    # steer penalty
    added_cost += STEER_COST * abs(steer)

    # steer change penalty
    added_cost += STEER_CHANGE_COST * abs(current.steer - steer)

    cost = current.cost + added_cost + arc_l

    # 根据该段控制空间采样曲线的起点，新建一个Node
    node = Node(
        x_ind,
        y_ind,
        yaw_ind,
        d,
        x_list,
        y_list,
        yaw_list,
        [d],
        parent_index=calc_index(current, config),
        cost=cost,
        steer=steer,
    )

    return node


def is_same_grid(n1, n2):
    if (
        n1.x_index == n2.x_index
        and n1.y_index == n2.y_index
        and n1.yaw_index == n2.yaw_index
    ):
        return True
    return False


# 输入：current:待拓展节点; goal:目标节点; ox,oy:实际地图; kd_tree:障碍物kd-tree;
# 输出：输出46条rs曲线中，没有碰撞，且rs曲线cost最小的一条曲线
# 功能：用rs曲线做一次从current到goal的one shot(原理见高飞视频教程)
def analytic_expansion(current, goal, ox, oy, kd_tree):
    start_x = current.x_list[-1]
    start_y = current.y_list[-1]
    start_yaw = current.yaw_list[-1]

    goal_x = goal.x_list[-1]
    goal_y = goal.y_list[-1]
    goal_yaw = goal.yaw_list[-1]

    # 计算出所有的46条rs曲线(每条path都已离散化)
    max_curvature = math.tan(MAX_STEER) / WB
    paths = rs.calc_paths(
        start_x,
        start_y,
        start_yaw,
        goal_x,
        goal_y,
        goal_yaw,
        max_curvature,
        step_size=MOTION_RESOLUTION,
    )

    if not paths:
        return None

    best_path, best = None, None

    for path in paths:
        # 将rs曲线上每个点转到质心，再取BUBBLE_R范围内的obs点，分别转换到车身坐标系下判碰.(False:有碰撞; True:无碰撞)
        if check_car_collision(path.x, path.y, path.yaw, ox, oy, kd_tree):
            # 计算每条rs曲线的代价，并记录代价最小的rs曲线
            cost = calc_rs_path_cost(path)
            if not best or best > cost:
                best = cost
                best_path = path

    return best_path


# 输入：current:待拓展节点; goal:目标节点; c:障碍物配置空间; ox,oy:实际地图; kd_tree:障碍物kd-tree;
# 输出：对当前节点current做one shot的结果。结果结构：(bool:是否shot成功; Node:成功后的新节点)
# 功能：用解析解拓展当前节点current(one shot的原理参照高飞视频)
def update_node_with_analytic_expansion(current, goal, c, ox, oy, kd_tree):
    # 用rs曲线做一次从current到goal的one shot(path是选出最优的一条rs曲线)
    path = analytic_expansion(current, goal, ox, oy, kd_tree)

    if path:
        if show_animation:
            plt.plot(path.x, path.y)
            plt.savefig(gif_creator.get_image_path())
        f_x = path.x[1:]
        f_y = path.y[1:]
        f_yaw = path.yaw[1:]

        # 当前点current的cost + rs曲线path的cost，作为f_cost
        f_cost = current.cost + calc_rs_path_cost(path)
        # 计算current的位姿(x_index,y_index,yaw_index)的唯一索引
        f_parent_index = calc_index(current, c)

        fd = []
        for d in path.directions[1:]:
            fd.append(d >= 0)

        f_steer = 0.0
        f_path = Node(
            current.x_index,
            current.y_index,
            current.yaw_index,
            current.direction,
            f_x,
            f_y,
            f_yaw,
            fd,
            cost=f_cost,
            parent_index=f_parent_index,
            steer=f_steer,
        )
        return True, f_path

    return False, None


# 输入：reed_shepp_path:rs曲线
# 输出：rs曲线的代价
# 功能：对rs曲线的“长度,正+反-变化,转向角大小steer,同向的LR变化”共4种元素施加惩罚
def calc_rs_path_cost(reed_shepp_path):
    cost = 0.0
    # 1.根据rs曲线长度计算惩罚(倒车部分惩罚系数5.0; 正向1.0)
    for length in reed_shepp_path.lengths:
        if length >= 0:  # forward
            cost += length
        else:  # back
            cost += abs(length) * BACK_COST

    # switch back penalty
    # 2.对rs曲线中每次“正+反-”行驶方向的变化施加一个100.0的惩罚
    for i in range(len(reed_shepp_path.lengths) - 1):
        # switch back
        if reed_shepp_path.lengths[i] * reed_shepp_path.lengths[i + 1] < 0.0:
            cost += SB_COST

    # steer penalty
    # 3.rs曲线中C段转向角的惩罚系数
    for course_type in reed_shepp_path.ctypes:
        if course_type != "S":  # curve
            cost += STEER_COST * abs(MAX_STEER)

    # == steer change penalty
    # calc steer profile
    # 4.rs曲线中正+反-同向，但是左L右R变化的惩罚系数
    n_ctypes = len(reed_shepp_path.ctypes)
    u_list = [0.0] * n_ctypes
    for i in range(n_ctypes):
        if reed_shepp_path.ctypes[i] == "R":
            u_list[i] = -MAX_STEER
        elif reed_shepp_path.ctypes[i] == "L":
            u_list[i] = MAX_STEER

    for i in range(len(reed_shepp_path.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(u_list[i + 1] - u_list[i])

    return cost


# 输入：始末点位姿，障碍物信息，xy方向的分辨率2.0m、yaw的分辨率15deg，
# 输出：从起点到终点的Path对象，包含各个离散点的cartisian信息(x,y,yaw,direction)与整体的cost
# 功能：hybrid A*搜索.
def hybrid_a_star_planning(start, goal, ox, oy, xy_resolution, yaw_resolution):
    """
    start: start node
    goal: goal node
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    xy_resolution: grid resolution [m]
    yaw_resolution: yaw angle resolution [rad]
    """

    # 对始末位姿yaw归一化：输入rad，输出[-pi, pi)
    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])
    tox, toy = ox[:], oy[:]

    # 将障碍物信息生成一个KDTree
    obstacle_kd_tree = cKDTree(np.vstack((tox, toy)).T)

    # 将地图按xy方向分辨率2.0m，yaw分辨率15度，进行栅格化
    config = Config(tox, toy, xy_resolution, yaw_resolution)

    # 建立hybrid A*图搜的始末节点Node
    start_node = Node(
        round(start[0] / xy_resolution),
        round(start[1] / xy_resolution),
        round(start[2] / yaw_resolution),
        True,
        [start[0]],
        [start[1]],
        [start[2]],
        [True],
        cost=0,
    )
    goal_node = Node(
        round(goal[0] / xy_resolution),
        round(goal[1] / xy_resolution),
        round(goal[2] / yaw_resolution),
        True,
        [goal[0]],
        [goal[1]],
        [goal[2]],
        [True],
    )

    # hybrid A*图搜的待访问点(openList), 已访问点(closeList).格式：(索引,Node)
    openList, closedList = {}, {} # 用"{}"初始化是dict；用"[]"初始化是list。

    # 用Dijkstra计算从goal_node到其它所有节点，考虑障碍物不考虑动力学的单源最短路径，作为距离启发函数h_dp(已计算启发值的节点的map)
    h_dp = calc_distance_heuristic(
        goal_node.x_list[-1], goal_node.y_list[-1], ox, oy, xy_resolution, BUBBLE_R
    )

    # 将起点start_node计算cost后加入openList
    pq = [] # list中存储元组tuple格式(cost,索引)
    openList[calc_index(start_node, config)] = start_node
    heapq.heappush(
        pq, (calc_cost(start_node, h_dp, config), calc_index(start_node, config))
    )
    final_path = None

    while True:
        if not openList:
            print("Error: Cannot find path, No open set")
            return [], [], []

        # 从openList中获取cost最小的，且未被访问过的点，继续访问
        cost, c_id = heapq.heappop(pq)
        if c_id in openList:
            current = openList.pop(c_id)
            closedList[c_id] = current
        else:
            continue

        if show_animation:  # pragma: no cover
            plt.plot(current.x_list[-1], current.y_list[-1], "xc")
            plt.savefig(gif_creator.get_image_path())
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            if len(closedList.keys()) % 10 == 0:
                plt.pause(0.001)

        # 用rs曲线对当前节点current做one shot(is_updated:是否更新成功; final_path:成功后生成的Node.)
        is_updated, final_path = update_node_with_analytic_expansion(
            current, goal_node, config, ox, oy, obstacle_kd_tree
        )

        # one-shot成功后就跳出循环
        if is_updated:
            print("path found")
            break

        # 对控制空间采样，并使用yield生成器返回采样生成neighbor无碰撞node
        for neighbor in get_neighbors(current, config, ox, oy, obstacle_kd_tree):
            # 求neighbor node在栅格地图中位姿(x_index,y_index,yaw_index)的索引
            neighbor_index = calc_index(neighbor, config)
            # 如果neighbor node已经在closedList中，则过滤；否则就进行松弛操作，并更新优先队列与openList。
            if neighbor_index in closedList:
                continue
            if (
                neighbor_index not in openList
                or openList[neighbor_index].cost > neighbor.cost
            ):
                heapq.heappush(pq, (calc_cost(neighbor, h_dp, config), neighbor_index))
                openList[neighbor_index] = neighbor

    # 生成从起点到终点的Path对象，包含各个离散点的cartisian信息(x,y,yaw,direction)与整体的cost
    path = get_final_path(closedList, final_path)
    return path


# 输入：n:待计算cost的节点; h_dp:启发函数(已计算启发值的节点的map); c:障碍物配置空间(已栅格化)
# 输出：n的代价cost=g+h（注：使用启发值h时用了工程上的trick，对h乘以5.0）
def calc_cost(n, h_dp, c):
    ind = (n.y_index - c.min_y) * c.x_w + (n.x_index - c.min_x)
    if ind not in h_dp:
        return n.cost + 999999999  # collision cost
    return n.cost + H_COST * h_dp[ind].cost # H_COST工程上的trick


# 输入：closed:已访问过节点的dict; goal_node:可以连接到goal的最后一个Node(最后一个Node中的采样曲线一定是rs曲线);
# 输出：生成Path对象，包含各个离散点的cartisian信息(x,y,yaw,direction)与整体的cost
def get_final_path(closed, goal_node):
    # 从后往前，即从goal开始一直找parent_index，生成一条逆向paht。最后再整体reverse。
    reversed_x, reversed_y, reversed_yaw = (
        list(reversed(goal_node.x_list)),
        list(reversed(goal_node.y_list)),
        list(reversed(goal_node.yaw_list)),
    )
    direction = list(reversed(goal_node.directions))
    nid = goal_node.parent_index
    final_cost = goal_node.cost

    while nid:
        n = closed[nid]
        reversed_x.extend(list(reversed(n.x_list)))
        reversed_y.extend(list(reversed(n.y_list)))
        reversed_yaw.extend(list(reversed(n.yaw_list)))
        direction.extend(list(reversed(n.directions)))

        nid = n.parent_index

    reversed_x = list(reversed(reversed_x))
    reversed_y = list(reversed(reversed_y))
    reversed_yaw = list(reversed(reversed_yaw))
    direction = list(reversed(direction))

    # adjust first direction
    direction[0] = direction[1]

    # 生成Path对象，包含各个离散点的cartisian信息(x,y,yaw,direction)与整体的cost
    path = Path(reversed_x, reversed_y, reversed_yaw, direction, final_cost)

    return path


# 输入：node:Node对象; c:障碍物配置空间;
# 输出：True:node在栅格地图中; False:node超出栅格地图
def verify_index(node, c):
    x_ind, y_ind = node.x_index, node.y_index
    if c.min_x <= x_ind <= c.max_x and c.min_y <= y_ind <= c.max_y:
        return True

    return False


# 输入：current:待拓展节点; c:障碍物配置空间;
# 输出：节点node(x_index,y_index,yaw_index)在栅格地图上的索引
# 功能：求node在栅格地图中位姿(x_index,y_index,yaw_index)的索引.(注：不管索引的具体数值，只要求每个位姿的索引都是唯一的)
def calc_index(node, c):
    ind = (
        (node.yaw_index - c.min_yaw) * c.x_w * c.y_w
        + (node.y_index - c.min_y) * c.x_w
        + (node.x_index - c.min_x)
    )

    if ind <= 0:
        print("Error(calc_index):", ind)

    return ind


# 建立地图border_x,border_y记录边界坐标;ox,oy记录障碍物占据的坐标(相当于都是1*1的栅格)
def construct_env_info():
    ox = []
    oy = []
    border_x = []
    border_y = []

    # road border.
    for i in range(0, 60, 1):
        border_x.append(i)
        border_y.append(0.0)
    for i in range(0, 60, 1):
        border_x.append(60.0)
        border_y.append(i)
    for i in range(0, 61, 1):
        border_x.append(i)
        border_y.append(60.0)
    for i in range(0, 61, 1):
        border_x.append(0.0)
        border_y.append(i)

    # Obstacle 1.
    for i in range(40, 55, 1):
        for j in range(5, 15, 1):
            ox.append(i)
            oy.append(j)

    # Obstacle 2.
    for i in range(40):
        for j in range(20, 25, 1):
            ox.append(j)
            oy.append(i)

    # Obstacle 3.
    for i in range(30):
        for j in range(40, 45, 1):
            ox.append(j)
            oy.append(60.0 - i)
    return border_x, border_y, ox, oy


def main():
    print("Start Hybrid A* planning")

    # Set Initial parameters(设置始末点的位姿)
    start = [10.0, 10.0, np.deg2rad(90.0)]
    goal = [50.0, 50.0, np.deg2rad(-90.0)]

    # construct environment info.
    # 建立地图border_x,border_y记录边界坐标;ox,oy记录障碍物占据的坐标(相当于都是1*1的栅格)
    border_x, border_y, ox, oy = construct_env_info()

    if show_animation:
        plt.plot(border_x, border_y, ".k", markersize=10)
        plt.plot(ox, oy, ".k")
        plt.plot(start[0], start[1], ".r", markersize=20)
        plt.plot(goal[0], goal[1], ".r", markersize=20)
        plot_arrow(start[0], start[1], start[2], fc="g")
        plot_arrow(goal[0], goal[1], goal[2])
        plt.savefig(gif_creator.get_image_path())
        plt.grid(True)
        plt.axis("equal")

    # 地图边界也当障碍物处理
    raw_ox = ox
    raw_oy = oy
    ox.extend(border_x)
    oy.extend(border_y)
    # 输入:障碍物信息，始末点位姿，xy方向的步长2.0m、yaw步长15deg，开始进行hybrid A*搜索.输出离散的路径点(位姿)
    # 生成:从start到goal的Path对象，包含各个离散点的cartisian信息(x,y,yaw,direction)与整体的cost
    path = hybrid_a_star_planning(
        start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION
    )

    x = path.x_list
    y = path.y_list
    yaw = path.yaw_list

    if show_animation:
        for i, (i_x, i_y, i_yaw) in enumerate(zip(x, y, yaw), start=1):
            if i % 5 == 0:
                plt.cla()
                plt.plot(border_x, border_y, ".k", markersize=10)
                plt.plot(ox, oy, ".k")
                plt.plot(start[0], start[1], ".r", markersize=20)
                plt.plot(goal[0], goal[1], ".b", markersize=20)
                plt.plot(x, y, "-r", label="Hybrid A* path")
                plt.grid(True)
                plt.axis("equal")
                plot_car(i_x, i_y, i_yaw)
                plt.pause(0.0001)
                plt.savefig(gif_creator.get_image_path())
    gif_creator.create_gif()
    plt.show()


if __name__ == "__main__":
    fig, ax = plt.subplots()
    gif_creator = GifCreator(file_path, fig, ax)
    main()
