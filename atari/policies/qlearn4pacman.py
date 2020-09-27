import gym
from gym import spaces
import numpy as np
import os
from collections import defaultdict

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DiscretizedObservationWapper(gym.ObservationWrapper) :
    def __init__(self, env, n_bins = 10, low = None, high = None) :
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        low  = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins+1) for l, h in zip(low, high)]
        self.observation_space = spaces.Discrete(n_bins ** len(low))

    def _covnert_to_one_number(self, digits):
        return sum([d * ((self.n_bins+1)**i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0] for x, bins in zip(observation.flatten(), self.val_bins)]
        return self._covnert_to_one_number(digits)

class Q_LearningPolicy :
    def __init__(self, env, alpha, alpha_decay, gamma, eps, eps_final, Q = None) :
        assert isinstance(env.action_space, spaces.Discrete)
        assert isinstance(env.observation_space, spaces.Discrete)
        self.env = env
        self.Q = Q
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.eps = eps
        self.eps_final = eps_final
        self.actions = range(self.env.action_space.n)

    # choose next action with eps-greedy
    def act(self, obs) :
        if np.random.random() < self.eps:
            return self.env.action_space.sample()
        
        qvals = {a : self.Q[obs, a] for a in self.actions}
        max_q = max(qvals.values())
        # select random one action to break a tie
        act_with_max_q = [a for a, q in qvals.items() if q == max_q]
        return np.random.choice(act_with_max_q)

    # update Q(s,a) with q-learning
    def update_Q(self, s, a, r, s_next, done) :
        max_q_next = max([self.Q[s_next, a] for a in self.actions])
        self.Q[s, a] += self.alpha * (r + self.gamma * max_q_next * (1.0-done) 
                                      - self.Q[s, a])

    def train(self, num_epsiode, warm_epsiode) :
        eps_desc = (self.eps - self.eps_final) / warm_epsiode
        rewards = []
        rewards_avg = []

        for epi in range(num_epsiode) :
            obs = self.env.reset()
            done  = False
            steps  = 0
            reward = 0
            while not done :
                # random choose action
                action = self.act(obs)
                # print(action)
                obs_next, r, done, info = self.env.step(action)
                reward += r
                self.update_Q(obs, action, r, obs_next, done)
                obs = obs_next
                steps += 1
            print("Episode terminates at {} steps!".format(steps))

            # update training param
            self.alpha *= self.alpha_decay
            if self.eps > self.eps_final :
                self.eps -= eps_desc

            # record result
            rewards.append(reward)
            rewards_avg.append(np.average(rewards[-50:]))
        return {'rwd' : rewards, 'rwd_avg' : rewards_avg}

# plot curve
def plot_learning_curve(filename, value_dict, xlabel='step'):
    # Plot step vs the mean(last 50 episodes' rewards)
    fig = plt.figure(figsize=(12, 4 * len(value_dict)))

    for i, (key, values) in enumerate(value_dict.items()):
        ax = fig.add_subplot(len(value_dict), 1, i + 1)
        ax.plot(range(len(values)), values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(key)
        ax.grid('k--', alpha=0.6)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    plt.tight_layout()
    os.makedirs(os.path.join(root, 'figs'), exist_ok=True)
    plt.savefig(os.path.join(root, 'figs', filename))

if __name__ == '__main__' :

    # env def. as playground
    g_env  = gym.make("MsPacman-v0")
    g_wenv = DiscretizedObservationWapper(g_env, n_bins = 8, 
                                      low=[-2.4, -2.0, -0.42, -3.5], high=[2.4, 2.0, 0.42, 3.5])

    # Q-Learning learner
    Q = defaultdict(float)
    gamma = 0.99 # discount for future
    alpha = 0.5 # soft param-update
    alpha_decay = 0.998
    eps = 1.0 # eps-greedy for exploration
    eps_final = 0.05

    # param for training iteration
    num_epsiode  = 1000
    warm_epsiode = 800

    learner = Q_LearningPolicy(g_wenv, alpha, alpha_decay, gamma, eps, eps_final, Q)
    reward_dict = learner.train(num_epsiode, warm_epsiode)
    plot_learning_curve('rewards_trace.png', reward_dict, xlabel = 'epsiode')
   
    g_wenv.close()