import os
import sys
from gym.spaces import Discrete, Box

import numpy as np
import tensorflow as tf
from gym.utils import colorize
from tensorflow.python.training.training_util import global_step

sys.path.append(os.path.join(os.path.dirname(__file__), "../lib"))
from utils import OUTPUT_ROOT

class TrainConfig :
    lrn_rate = 0.001
    lrn_rate_decay = 1.0
    epsilon = 1.0
    epsilon_final = 0.01
    memmory_capacity = 100000
    target_update_every_steps = 100
    total_episodes = 500
    warmup_episodes = 450
    log_every_episodes = 100

class TFModelBase :
    def __init__(self, name, session_config = None) :
        self._saver  = None
        self._writer = None
        self._name   = name
        self._session = None
        if session_config is None :
            session_config = {
                'allow_soft_placement' : True, 
                'intra_op_parallelism_threads' : 8, 
                'inter_op_parallelism_threads' : 4
            }
        self._config = session_config

    def save_checkpoint(self, step = None) :
        print(colorize(" [*] Saving checkpoints...", "green"))
        ckpt_fn = os.path.join(self.ckpt_dir, self._name)
        self.saver.save(self.session, ckpt_fn, global_step = step)
    
    def load_checkpoint(self) :
        ckpt_path = tf.train.latest_checkpoint(self.ckpt_dir)
        print(colorize(" [*] Load checkpoint from [{}]/[{}]".format(self.ckpt_dir, ckpt_path), "green"))
        if ckpt_path :
            self.saver.restore(self.session, ckpt_path)
            print(colorize(" [*] Load Success!", "green"))
            return True
        else :
            print(colorize(" [!] Load Failed!", "red"))
            return False

    def _make_dir(self, dir_name) :
        path = os.path.join(OUTPUT_ROOT, self._name, dir_name)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def log_dir(self) :
        return self._make_dir('log')

    @property
    def ckpt_dir(self) :
        return self._make_dir('ckpt')

    @property
    def model_dir(self) :
        return self._make_dir('model')

    @property
    def tb_dir(self) :
        return self._make_dir('tensorboard')

    @property
    def figs_dir(self) :
        return self._make_dir('figs')

    @property
    def saver(self) :
        if self._saver is None :
            self._saver = tf.train.Saver(max_to_keep=5)
        return self._saver

    @property
    def writer(self) :
        if self._writer is None :
            self._writer = tf.summary.FileWriter(self.tb_dir, self.session.graph)
        return self._writer

    @property
    def session(self) :
        if self._session is None :
            config = tf.ConfigProto(**self._config)
            self._session = tf.Session(config = config)
        return self._session

class Policy(TFModelBase) :
    def __init__(self, env, name, tf_config = None,
                 training=True, gamma=0.99, deterministic=False):
        TFModelBase.__init__(self, name, tf_config)
        self.env = env
        self.gamma = gamma
        self.training = training
        self.name = name

        if deterministic :
            np.random.seed(1)
            tf.set_random_seed(1)

    @property
    def action_size(self) :
        if isinstance(self.env.action_space, Discrete) :
            return self.env.action_space.n
        return None

    @property
    def action_dim(self) :
        if isinstance(self.env.action_space, Box) :
            return list(self.env.action_space.shape)
        return []

    @property
    def state_dim(self) :
        return list(self.env.observation_space.shape)

    @property
    def state_size(self) :
        if isinstance(self.env.observation_space, Box) :
            return np.prod(self.env.observation_space.shape)
        return None

    def obs_to_input(self, obs) :
        return obs.flatten()

    def act(self, state, **kwargs) :
        pass

    def create_networks(self, **kwargs) :
        pass

    def def_loss_and_optimizer(self, **kwargs) :
        pass

    def summary(self, **kwargs) :
        pass

    # 4 steps for building networks & init.
    def build(self, **kwargs) :
        print(colorize("[Build-1/4]: create networks.", "green"))
        self.create_networks()
        print(colorize("[Build-2/4]: define loss and optimizer.", "green"))
        self.def_loss_and_optimizer()
        print(colorize("[Build-3/4]: summary on dashboard.", "green"))
        self.summary()
        print(colorize("[Build-4/4]: initialization.", "green"))
        self.initialize()

    def train(self, *args, **kwargs) :
        pass

    def eval(self, n_episodes) :
        reward_hist = []
        reward = 0

        for i in range(n_episodes) :
            obs = self.env.reset()
            done = False
            while not done :
                a = self.act(obs)
                new_obs, r, done, info = self.env.step(a)
                self.env.render()
                reward += r
                obs = new_obs
            reward_hist.append(reward)
            reward = 0
        print("Average reward over {} episodes: {:.4f}".format(n_episodes, np.mean(reward_hist)))