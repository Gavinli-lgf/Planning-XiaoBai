import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import sys
import pathlib

file_path = pathlib.Path(__file__)
root_dir = file_path.parent.parent.parent
sys.path.append(str(root_dir))

import copy
from common.geometry import *
from common.plot_util import *
from common.gif_creator import *
from common.common_util import *

# hyper parameter, and they can be adjusted.
N_SAMPLE = 500  # number of sample_points(采样点的数量)
N_KNN = 10  # number of edge from one sampled point(限制PRM中，每个点最多与另外10个点相连)
MAX_EDGE_LEN = 30.0  # [m] Maximum edge length

show_animation = True


# Node class for dijkstra search
class Node:
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


# 输入：输入参数解释如下
# 输出：规划路径坐标rx,ry
# 功能：prm规划的主体功能，生成障碍物的KDTree,学习阶段,查询阶段,生成prm规划结果
def prm_planning(
    start_x,
    start_y,
    goal_x,
    goal_y,
    obstacle_x_list,
    obstacle_y_list,
    robot_radius,
    *,
    rng=None
):
    """
    Run probabilistic road map planning

    :param start_x: start x position
    :param start_y: start y position
    :param goal_x: goal x position
    :param goal_y: goal y position
    :param obstacle_x_list: obstacle x positions(包括边界)
    :param obstacle_y_list: obstacle y positions(包括边界)
    :param robot_radius: robot radius
    :param rng: (Optional) Random generator
    :return:
    """
    # 1. 生成障碍物(包括边界)的KDTree
    obstacle_kd_tree = KDTree(np.vstack((obstacle_x_list, obstacle_y_list)).T)

    # 2. 均匀随机采样，丢弃与障碍物碰撞的点
    # Uniform random sampling, and discard points that collide with obstacles.
    sample_x, sample_y = sample_points(
        start_x,
        start_y,
        goal_x,
        goal_y,
        robot_radius,
        obstacle_x_list,
        obstacle_y_list,
        obstacle_kd_tree,
        rng,
    )
    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    # 3. 生成PRM的邻接表，邻接表中存储了距离每个采样点最近的前N_KNN个无碰撞的边
    road_map = generate_road_map(sample_x, sample_y, robot_radius, obstacle_kd_tree)

    # 4. 在路图中使用dijkstra搜索得到start到goal的路径rx,ry
    rx, ry = dijkstra_planning(
        start_x, start_y, goal_x, goal_y, road_map, sample_x, sample_y
    )

    return rx, ry


# 输入：sx, sy, gx, gy, rr, obstacle_kd_tree:起点坐标，终点坐标，obs尺寸半径，obs的KDTree
# 输出：True:有碰撞; False:无碰撞;
# 功能：对(sx,sy)到(gx,gy)的直线，以rr的间隔采样，判断每个采样点与obs是否有碰撞(包括距离obs<rr)
def is_collision(sx, sy, gx, gy, rr, obstacle_kd_tree):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN:
        return True

    D = rr
    n_step = round(d / D)

    # 增量式取点，判断边与obs是否碰撞(包括距离obs<rr)
    for i in range(n_step):
        dist, _ = obstacle_kd_tree.query([x, y])
        if dist <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    dist, _ = obstacle_kd_tree.query([gx, gy])
    if dist <= rr:
        return True  # collision

    return False  # OK


# 输入：sample_x, sample_y, robot_radius, obstacle_kd_tree
# 输出：road_map
# 功能：生成PRM的邻接表，邻接表中存储了距离每个采样点最近的前N_KNN个无碰撞的边
def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree):
    """
    Road map generation

    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    robot_radius: Robot Radius[m]
    obstacle_kd_tree: KDTree object of obstacles
    """

    # 生成的PRM路图以临界表的方式存储
    road_map = []
    n_sample = len(sample_x)
    # 生成采样点的KDTree
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)

    for i, ix, iy in zip(range(n_sample), sample_x, sample_y):
        # 在采样点的KDTree中查找距离(ix,iy)最近的前n_sample个点。且查询结果"dists(距离),indexes(索引)"按照由近到远的方式排列
        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
        # 邻接表中点(ix,iy)的所有出度点
        edge_id = [] 

        # 查找距离(ix,iy)最近的前N_KNN个无碰撞的边，并将出度存储在edge_id中
        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]

            # 对(ix,iy)到(nx,ny)的直线，以rr的间隔采样，判断每个采样点与obs是否有碰撞(包括距离obs<rr)
            if not is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree):
                edge_id.append(indexes[ii])

            # PRM中每个点最多与其他N_KNN个点连接生成边
            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    return road_map


def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y):
    """
    s_x: start x position [m]
    s_y: start y position [m]
    goal_x: goal x position [m]
    goal_y: goal y position [m]
    obstacle_x_list: x position list of Obstacles [m]
    obstacle_y_list: y position list of Obstacles [m]
    robot_radius: robot radius [m]
    road_map: ??? [m]
    sample_x: ??? [m]
    sample_y: ??? [m]

    @return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found
    """

    start_node = Node(sx, sy, 0.0, -1)
    goal_node = Node(gx, gy, 0.0, -1)

    open_set, closed_set = dict(), dict()
    open_set[len(road_map) - 2] = start_node

    path_found = True

    while True:
        if not open_set:
            print("Cannot find path")
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        # show graph
        if show_animation and len(closed_set.keys()) % 2 == 0:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            plt.plot(current.x, current.y, "xg")
            plt.savefig(gif_creator.get_image_path())
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break

        # Remove the item from the open set
        del open_set[c_id]
        # Add it to the closed set
        closed_set[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)
            node = Node(sample_x[n_id], sample_y[n_id], current.cost + d, c_id)

            if n_id in closed_set:
                continue
            # Otherwise if it is already in the open set
            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node

    if path_found is False:
        return [], []

    # generate final course
    rx, ry = [goal_node.x], [goal_node.y]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index

    return rx, ry


def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover

    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]

            plt.plot([sample_x[i], sample_x[ind]], [sample_y[i], sample_y[ind]], "-k")


# 功能：均匀随机采样，丢弃与障碍物碰撞的点("与obs距离<robot_radius"的也叫碰撞)
# 输入：start_x, start_y, goal_x, goal_y, robot_radius, obstacle_x_list, obstacle_y_list, obstacle_kd_tree, rng
# 输出：采样点sample_x,sample_y
def sample_points(sx, sy, gx, gy, rr, ox, oy, obstacle_kd_tree, rng):
    max_x = max(ox)
    max_y = max(oy)
    min_x = min(ox)
    min_y = min(oy)

    sample_x, sample_y = [], []

    # lgf推断np.random.default_rng()生成0.0~1.0之间的随机数
    if rng is None:
        rng = np.random.default_rng()

    # 采样与obs无碰撞的点个数为N_SAMPLE个
    while len(sample_x) <= N_SAMPLE:
        tx = (rng.random() * (max_x - min_x)) + min_x
        ty = (rng.random() * (max_y - min_y)) + min_y

        dist, index = obstacle_kd_tree.query([tx, ty])

        if dist >= rr:
            sample_x.append(tx)
            sample_y.append(ty)

    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y


# 功能：构建地图与障碍物信息。被地图边界、障碍物所占据的区域，分别记录在border_x,border_y与ox,oy中
# (与图搜的区别是:不用栅格化)
def construct_env_info():
    ox = []
    oy = []
    border_x = []
    border_y = []

    # road border.(60m*60m的地图)
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


# Probabilistic Road Map实现主体
# 输入：可选项 rng(随机生成器)
def prm(rng=None):
    print("Begin to run the prm!!!")

    # 1. 定义已知量:起始点，机器人尺寸
    # start and goal position.
    start_x = 10.0  # [m]
    start_y = 10.0  # [m]
    goal_x = 50.0  # [m]
    goal_y = 50.0  # [m]
    robot_size = 5.0  # [m]

    # 2. 构建地图与障碍物信息(与图搜的区别是:不用栅格化)
    # construct environment info.
    border_x, border_y, ox, oy = construct_env_info()

    if show_animation:
        plt.plot(border_x, border_y, ".g", markersize=10)
        plt.plot(ox, oy, ".k")
        plt.plot(start_x, start_y, ".r", markersize=20)
        plt.plot(goal_x, goal_y, ".r", markersize=20)
        plt.grid(True)
        plt.axis("equal")

    # 3. prm(概率路图规划),返回规划结果rx,ry
    # run the prm planning.
    ox.extend(border_x)
    oy.extend(border_y)
    rx, ry = prm_planning(start_x, start_y, goal_x, goal_y, ox, oy, robot_size, rng=rng)

    # assert rx, 'Cannot found path'

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.savefig(gif_creator.get_image_path())
        plt.pause(0.001)
        gif_creator.create_gif()
        plt.show()


if __name__ == "__main__":
    fig, ax = plt.subplots()
    gif_creator = GifCreator(file_path, fig, ax)
    prm()
