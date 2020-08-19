import sys
import numpy as np
from environment import MountainCar


class Q_Learning:
    def __init__(self, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, learning_rate):
        self.mode = mode
        self.weight_out = weight_out
        self.returns_out = returns_out
        self.episodes = episodes
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.mc = None
        self.w = None
        self.b = None

        self.rolling_mean_25 = np.array([])
        self.total_rewards_list = np.array([])

    def initialize_mc(self):
        self.mc = MountainCar(self.mode)

    def initialize_weights(self):
        self.w = np.zeros((self.mc.state_space, self.mc.action_space))
        self.b = 0

    def write_weights_output(self):
        f_weight_out = open(self.weight_out, "w+")
        f_weight_out.write("{0}\n".format(self.b))
        for w in self.w.flat:
            f_weight_out.write("{0}\n".format(w))
        f_weight_out.close()

    @staticmethod
    def qsaw(state, action, weight):
        w = weight[:, action]
        product = 0
        for key, value in state.items():
            product += w[key] * value
        return product

    def qvalues_calculation(self, state):
        return [self.qsaw(state, i, self.w) + self.b for i in range(self.mc.action_space)]

    def next_action(self, state):
        q = self.qvalues_calculation(state)
        return q.index(np.max(q)) if self.epsilon == 0 or np.random.uniform(0, 1) >= self.epsilon else np.random.choice((0, 1, 2))

    def train(self):
        f_returns_out = open(self.returns_out, "w+")
        for cur_episode in range(self.episodes):
            cur_state = self.mc.reset()
            done = False
            cur_iteration = 1
            total_reward = 0
            while not done and cur_iteration <= self.max_iterations:
                # get an action to take
                next_action = self.next_action(cur_state)
                # take a step
                next_state, reward, done = self.mc.step(next_action)

                qsaw = self.qsaw(cur_state, next_action, self.w) + self.b
                max_qsaw = np.max(self.qvalues_calculation(next_state))

                # train the weights
                for i, v in cur_state.items():
                    self.w[i][next_action] -= self.learning_rate * (qsaw - (reward + self.gamma * max_qsaw)) * cur_state[i]
                self.b -= self.learning_rate * (qsaw - (reward + self.gamma * max_qsaw))

                # make current state = next state
                cur_state = next_state
                # update the total reward
                total_reward += reward
                cur_iteration += 1
            # print("Episode {0}, Total Reward {1}".format(cur_episode, total_reward))
            print(total_reward)
            f_returns_out.write("{0}\n".format(total_reward))
            self.total_rewards_list = np.append(self.total_rewards_list, total_reward)
            if cur_episode % 25 == 0:
                self.rolling_mean_25 = np.append(self.rolling_mean_25, np.average(self.total_rewards_list[len(self.total_rewards_list) - 25: len(self.total_rewards_list)]))

        f_returns_out.close()
        self.write_weights_output()
        #return self.total_rewards_list, self.rolling_mean_25

def main(args):
    # mode = args[1]
    # weight_out = args[2]
    # returns_out = args[3]
    # episodes = int(args[4])
    # max_iterations = int(args[5])
    # epsilon = np.float64(args[6])
    # gamma = np.float64(args[7])
    # learning_rate = np.float64(args[8])

    mode = "tile"
    weight_out = "weight.out"
    returns_out = "returns.out"
    episodes = 400
    max_iterations = 200
    epsilon = 0.05
    gamma = 0.99
    learning_rate = 0.00005

    Q = Q_Learning(mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, learning_rate)
    Q.initialize_mc()
    Q.initialize_weights()
    Q.train()

    Q.total_rewards_list.tofile("total_rewards_list.txt", sep=",", format="%s")
    Q.rolling_mean_25.tofile("rolling_mean_25.txt", sep=",", format="%s")
    #print(Q.b)
    #print(Q.w)

if __name__ == "__main__":
    main(sys.argv)