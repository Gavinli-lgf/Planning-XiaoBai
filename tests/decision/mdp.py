import numpy as np

# 1. Define the state space.(状态空间)
states = np.arange(25)

# 2. Define the action space:(动作空间)
# 0 --> top, 1 --> right, 2 --> bottom, 3 --> left.
actions = np.arange(4)

# 3. Define the rewards.(奖励:默认都是-1.0,宝藏序号22取0.0)
rewards = np.ones(25) * -1.0
rewards[22] = 0.0

# 4. Define the discount factor.(折扣因子)
gamma = 0.9


# 5. Define the transition matrix:(状态转移矩阵)
# input: current state, and action(输入：当前状态，动作)
# output: transition probability, next state, and reward.(输出：转移概率，转移后的状态，奖励)
def p_state_reward(state, action):
    # action: top --> 0(向上)
    if action == 0:
        if state in range(5):   #状态0~4时执行"向上"动作，依然回到当前状态
            return (1.0, state, -1.0)
        else:   # 其它状态时执行"向上"动作后的输出
            return (1, state - 5, -1)

    # action: down --> 2（向下）
    if action == 2:
        if state in range(20, 25):
            return (1.0, state, -1)
        elif state == 17:   # 从17向下找到宝藏后的奖励是0.0
            return (1.0, state + 5, 0.0)
        else:
            return (1.0, state + 5, -1.0)

    # action: left --> 3（向左）
    if action == 3:
        if state in range(0, 25, 5):
            return (1.0, state, -1.0)
        elif state == 23:   # 从23向左找到宝藏后的奖励是0.0
            return (1.0, state - 1, 0.0)
        else:
            return (1.0, state - 1, -1.0)

    # action: right --> 1（向右）
    if action == 1:
        if state in range(4, 29, 5):
            return (1.0, state, -1.0)
        elif state == 21:   # 从21向右找到宝藏后的奖励是0.0
            return (1.0, state + 1, 0.0)
        else:
            return (1.0, state + 1, -1.0)


# 6. Solver with policy iteration.(定义策略迭代的2个步骤：)
# 6.1 Policy evaluation: calculate the state value in given strategy.(策略评估)
# 输入：策略policy，折扣因子gamma； 输出：收敛后的状态价值列表
def compute_value_function(policy, gamma):
    # Set threshold.(状态价值是否收敛的阈值)
    threshold = 1e-10

    # Initialize the state value.(初始每个状态的价值为0.0)
    value_table = np.zeros(len(states))

    # Begin to iterate.
    while True:
        # Create the state value table in each iteration.(计算每次迭代后新的状态价值)
        update_value_table = np.copy(value_table)

        # Loop for each state.
        for state in states:    # 遍历每个状态
            # Get the action.(选择当前策略下，当前状态所对应的动作)
            action = policy[state]

            # Calculate the transition probability, next state, and reward.
            # (根据状态转移矩阵，获取当前状态state下，执行动作action，得到的"转移概率，下一个状态，奖励")
            prob, next_state, reward = p_state_reward(state, action)

            # Calculate the state value.根据V(s)状态价值公式，更新当前状态的状态价值
            value_table[state] = reward + gamma * prob * update_value_table[next_state]

        # If the state_valueis converged, break the loop.
        # (如果状态价值收敛，则break跳出循环)
        if np.sum(np.fabs(value_table - update_value_table)) < threshold:
            break

    return value_table


# 6.2 Policy improvement: improve the current policy based on the state value.(策略提升)
# 输入：策略评估中得到的收敛后的状态价值列表value_table， 折扣系数gamma； 输出：该状态价值列表下的最优策略；
def next_best_policy(value_table, gamma):
    # Initialize the policy.(创建大小与状态个数相等的空数组，记录改进后的策略)
    policy = np.zeros(len(states))

    # Loop for each state.(遍历每个状态)
    for state in states:
        # Initialize the action value.(创建大小该状态下动作个数4相等的空数组，记录每个动作的动作价值)
        action_table = np.zeros(len(actions))

        # Loop for each action.(遍历4个action中的每一个)
        for action in actions:
            # Calculate the transition probability, next state, and reward.
            # (根据状态转移矩阵，获取当前状态state下，执行动作action，得到的"转移概率，下一个状态，奖励")
            prob, next_state, reward = p_state_reward(state, action)

            # Calculate the action value.(计算该状态下该动作的动作价值)
            action_table[action] = prob * (reward + gamma * value_table[next_state])

        # Choose the best action.(获取动作价值列表action_table中最大动作价值的索引，该索引值也是动作)
        # 即，选取动作价值最大的动作，作为该状态时的最优策略
        policy[state] = np.argmax(action_table)

    return policy


# 6.3 Construct policy iteration function.(策略迭代算法的整体实现)
def policy_iteration(random_policy, gamma, n):
    # Begin to iterate.(最多迭代1000次)
    for i in range(n):
        # Policy evaluation.(策略评估：计算得到收敛后的状态价值列表)
        new_value_function = compute_value_function(random_policy, gamma)

        # Policy improvement.(策略改善:选取动作价值最大的动作更新策略)
        new_policy = next_best_policy(new_value_function, gamma)

        # Judge the current policy.(判断策略是否收敛，如果收敛就break)
        if np.all(random_policy == new_policy):
            print("End to iterate, and num is: %d" % (i + 1))
            break

        # Replace the optimal policy.(弱策略还没收敛时，每次更新为新的策略,用于下次策略评估)
        random_policy = new_policy

    return new_policy


# 7. Solver with the value iteration.(价值迭代)
def value_iteration(value_table, gamma, n):
    value_table = np.zeros(len(states)) # 初始化状态价值列表（默认都取0.0）
    threshold = 1e-20   # 定义状态价值收敛阈值
    policy = np.zeros(len(states))  # 定义策略列表(不使用初始值)

    # Begin to iterate.（最多迭代1000次）
    for i in range(n):
        update_value_table = np.copy(value_table) # 记录上一次状态价值列表(用于和本次状态价值列表比较，是否收敛)

        # Loop for each state.(遍历每个状态)
        for state in states:
            action_value = np.zeros(len(actions)) # 定义该状态下每个策略对应所有动作的动作价值列表

            # Loop for each action.(变量该状态下对应策略中的每个动作)
            for action in actions:

                # Calculate the transition probability, next state, and reward.
                # 根据状态转移矩阵，得到状态state下执行动作action多得到的转移概率trans_prob,下一个状态next_state,奖励reward
                trans_prob, next_state, reward = p_state_reward(state, action)

                # Calculate the action value.(计算动作action的动作价值)
                action_value[action] = (
                    reward + gamma * trans_prob * update_value_table[next_state]
                )

            # Update the state value table.(获取该状态下策略中所有动作的最大动作价值为该状态的状态价值)
            value_table[state] = max(action_value)

            # Record the optimal policy(获取该状态下策略中动作价值最大的动作为该状态的最优策略，即贪心的将概率置1了)
            policy[state] = np.argmax(action_value)

        ## End to iterate.(若相邻两次价值列表之差小于阈值，则停止循环)
        if np.sum((np.fabs(update_value_table - value_table))) <= threshold:
            print("End to iterate, and num is: %d" % (i + 1))
            break

    return policy


kUseValueIteration = True


def main():
    # Set iteration num.(设置最大迭代次数,1000次)
    n = 1000
    if kUseValueIteration:  # 价值迭代
        value_table = np.zeros(len(states)) # 初始化一个默认价值列表(都取0)
        best_policy = value_iteration(value_table, gamma, n) # 得到收敛后的最佳策略
    else:   # 策略迭代
        random_policy = 2 * np.ones(len(states)) # 初始化一个随机策略值都是2，即都"向下"
        best_policy = policy_iteration(random_policy, gamma, n) # 得到收敛后的最佳策略

    print("best policy is: ", best_policy)

    # Find the optimal route.(根据最佳策略，得到最佳行动路线)
    best_route = [0]    # 定义最佳行动路径列表(同时也记录了初始路径点0)
    next_state = 0      # 临时变量，记录每一步最优策略得到的状态;(同时也定义起始状态0)
    while True:

        # Solve the next state to which the optimal action is transferred through the best strategy in the current state
        # 计算通过当前状态next_state的最优策略best_policy[next_state]所对应的动作，可以达到的下一个状态next_state
        _, next_state, _ = p_state_reward(next_state, best_policy[next_state])

        # Add the next state to the best route list(将执行最优策略所得到的每一个状态，顺序加入best_route)
        best_route.append(next_state)

        # Transfer to termination state, stop loop(转移到终止状态时，停止循环)
        if next_state == 22:
            break

    print("The best route is: ", best_route)


if __name__ == "__main__":
    main()
