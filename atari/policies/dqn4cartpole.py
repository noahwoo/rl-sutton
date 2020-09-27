import gym
from gym import spaces
from gym.envs import kwargs
from gym.utils import colorize
import numpy as np
import os
import sys
from collections import defaultdict, deque, namedtuple
from numpy.lib.arraysetops import isin
from numpy.lib.index_tricks import AxisConcatenator
import tensorflow as tf
import itertools

import tensorflow

sys.path.append(os.path.join(os.path.dirname(__file__), "../lib"))
from utils import ReplayMemory
from utils import Transition
from utils import plot_learning_curve
from tf_utils import dense_nn

# policy module
from base import Policy
from base import TrainConfig

class DQN_Policy(Policy) :
    def __init__(self, name, env, gamma, batch_size, layer_sizes, training = True, tf_config = None) :
        Policy.__init__(self, env, name, tf_config, training, gamma)
        # network 
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes

    def create_networks(self, **kwargs) :
        print("ENTER: [{}]".format(sys._getframe(  ).f_code.co_name))
        # tf variables
        self.states       = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')
        self.next_states  = tf.placeholder(tf.float32, shape=(None, self.state_size), name='next_state')
        self.actions      = tf.placeholder(tf.int32, shape=(None, ), name='action')
        self.next_actions = tf.placeholder(tf.int32, shape=(None, ), name='next_action')
        self.rewards      = tf.placeholder(tf.float32, shape=(None, ), name='reward')
        self.dones        = tf.placeholder(tf.float32, shape=(None, ), name='done')

        # Q-networks and target networks
        self.Q        = dense_nn(self.states, self.layer_sizes + [self.action_size], 
                                      name='q_primary')
        self.Q_target = dense_nn(self.next_states, self.layer_sizes + [self.action_size], 
                                      name='q_target')

        self.Q_var        = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "q_primary")
        self.Q_target_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "q_target")
        assert len(self.Q_var) == len(self.Q_target_var)
        print("OUT: [{}]".format(sys._getframe(  ).f_code.co_name))

    def def_loss_and_optimizer(self, **kwargs) :
        print("ENTER: [{}]".format(sys._getframe(  ).f_code.co_name))
        # loss and optimizer
        # Q: None * action_size
        self.action_selected_by_q = tf.argmax(self.Q, axis = -1, name = 'action_selected')
        action_one_hot = tf.one_hot(self.action_selected_by_q, self.action_size, 1.0, 0.0, 
                                    name='action_one_hot')

        # q(s,a) = pred : None * 1
        self.pred  = tf.reduce_sum(self.Q * action_one_hot, reduction_indices = -1, name = 'q_acted')
        # target = r + gamma * max_a' q(s', a'): None * action_size
        max_q_next = tf.reduce_max(self.Q_target, axis = -1)
        self.target = self.rewards + (1 - self.dones) * self.gamma * max_q_next
        # loss = mean((target - q(s,a))^2)
        # `stop_gradient' stop the loss gradient from back-prop to Q_target var., updating by lag copy only
        self.loss = tf.reduce_mean(tf.square(self.pred - tf.stop_gradient(self.target)), name = 'mse_loss')
        # adam optimizer
        self.learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, name='adam_optim')
        print("OUT: [{}]".format(sys._getframe(  ).f_code.co_name))

    def summary(self, **kwargs) :
        print("ENTER: [{}]".format(sys._getframe(  ).f_code.co_name))
        # summary for tensorboard
        with tf.variable_scope('summary') :
            # Q summary in histogram
            q_summary = []
            avg_q = tf.reduce_mean(self.Q, 0) # action_size
            for a in range(self.action_size) :
                q_summary.append(tf.summary.histogram('q/%s' % a, avg_q[a]))
            self.q_summary = tf.summary.merge(q_summary, 'q_summary')
            # pred and target
            self.q_target_summary = tf.summary.histogram('batch/target', self.target)
            self.q_pred_summary = tf.summary.histogram('batch/pred', self.pred)
            self.loss_summary = tf.summary.scalar('loss', self.loss)
            # episode reward
            self.epi_reward = tf.placeholder(tf.float32, name = 'episode_reward')
            self.epi_reward_summary = tf.summary.scalar('episode_reward', self.epi_reward)
            self.merged_summary = tf.summary.merge_all(key = tf.GraphKeys.SUMMARIES)
        print("OUT: [{}]".format(sys._getframe(  ).f_code.co_name))

    def initialize(self, **kwargs) :
        # do global initializer
        self.session.run(tf.global_variables_initializer())
        self._update_target_q_net()
 
    def _update_target_q_net(self, hard = True, tau = 0.5) :
        if hard == True :
            self.session.run([v_t.assign(v) 
                              for v_t, v in zip(self.Q_target_var, self.Q_var)])
        else:
            self.session.run([v_t.assign(v_t * (1.0 - tau) + v * tau) 
                              for v_t, v in zip(self.Q_target_var, self.Q_var)])

    def act(self, state, eps = 0.1) :
        if self.training and np.random.random() < eps :
            return self.env.action_space.sample()
        with self.session.as_default() :
            return self.action_selected_by_q.eval({self.states : [state]})[-1]

    def train(self, config : TrainConfig) :
        # experience replay memory
        replay_mem = ReplayMemory(config.memmory_capacity)
        # reward history
        reward = 0
        reward_history = []
        reward_avg = []
        # learning rate related
        alpha = config.lrn_rate
        eps = config.epsilon
        eps_delta = (config.epsilon - config.epsilon_final) / config.warmup_episodes

        step = 0
        for epi in range(config.total_episodes) :
            obs  = self.env.reset()
            done = False
            traj = []
            reward = 0
            while not done :
                # random choose action with epsilon-greedy
                action = self.act(obs, eps)
                obs_next, r, done, info = self.env.step(action)
                reward += r
                step += 1
                # record trajectories
                traj.append(Transition(obs.flatten(), action,
                                       r, obs_next.flatten(), done))
                obs = obs_next
                if replay_mem.size < self.batch_size :
                    continue
                # update q networks with mini-batch replay samples
                batch_data = replay_mem.sample(self.batch_size)
                feed_dict = {
                    self.learning_rate : alpha,
                    self.states : batch_data['s'],
                    self.actions : batch_data['a'],
                    self.rewards : batch_data['r'],
                    self.next_states : batch_data['s_next'],
                    self.dones : batch_data['done'],
                    self.epi_reward : reward_history[-1]
                }
                _, q, q_target, loss, summary = self.session.run([self.optimizer, 
                                                                  self.Q, self.Q_target, self.loss, self.merged_summary],
                                                                 feed_dict)
                # update target q networks hardly
                if step % config.target_update_every_steps == 0 :
                    self._update_target_q_net()
                self.writer.add_summary(summary)

            replay_mem.add(traj)
            # one episode done
            reward_history.append(reward)
            reward_avg.append(np.mean(reward_history[-10:]))

            # update training param
            alpha *= config.lrn_rate_decay
            if eps > config.epsilon_final :
                eps -= eps_delta
            
            # report progress
            # if reward_history and config.log_every_episodes and epi % config.log_every_episodes == 0 :
            print(
                   "[episodes:{}/step:{}], best:{}, avg:{:.2f}:{}, lrn_rate:{:.4f}, eps:{:.4f}".format(
                       epi, step, np.max(reward_history), np.mean(reward_history[-10:]), 
                       reward_history[-5:], alpha, eps)
            )
        
        self.save_checkpoint(step=step)
        print("[FINAL] episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)))
        return {'rwd' : reward_history, 'rwd_avg' : reward_avg}

if __name__ == '__main__' :
    genv = gym.make('CartPole-v1')
    obs_size = np.prod(list(genv.observation_space.shape))
    print("obs_size:{}, obs_shape:{}".format(obs_size, genv.observation_space.shape))
    
    # init. TF session
    tf_sess_config = {
                'allow_soft_placement': True,
                'intra_op_parallelism_threads': 8,
                'inter_op_parallelism_threads': 4,
            }

    training = False
    dqn = DQN_Policy("dqn_cartpole", genv, 0.99, 64, [32, 32], training, tf_sess_config)
    if training : 
        # dqn training
        config = TrainConfig()
        dqn.build()
        reward_dict = dqn.train(config)
        plot_learning_curve(os.path.join(dqn.figs_dir, 'dqn_train.png'), 
                            reward_dict, xlabel='step')
    else :
        dqn.build()
        if dqn.load_checkpoint() :
            dqn.eval(5)