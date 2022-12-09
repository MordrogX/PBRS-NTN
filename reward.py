import gym
import numpy as np
import pandas as pd
import torch
import os
from ntnmodel import LitModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.DataFrame(columns=['old_state', 'new_state', 'reward'])

class custom_obs(gym.ObservationWrapper):
    current_state = []
    def __init__(self, env):
        super().__init__(env)
        

    def observation(self, obs):
        custom_obs.current_state = obs

        return obs
def data_preprocessing(data):
    scale = StandardScaler().fit(data)
    data = scale.transform(data)
    norm = MinMaxScaler().fit(data)
    data = norm.transform(data)
    return data

class custom_reward(gym.RewardWrapper):
    i = 4096
    k = 0
    obs_state = [0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18]
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.env = env
        self.prep = data_preprocessing
        initial_state = self.env.reset()
        self.o_state = initial_state
        resized_initial_state = [initial_state[i] for i in iter(custom_reward.obs_state)]
        self.old_theta = 0
        self.final_state = [1.0, 0.0, 0.0, 0.0, 0.0, 10000.0, 0, 0, 0, 0, 0]
        mag = [self.final_state[i] - resized_initial_state[i] for i in range(0, 11)]
        self.old_mag = np.linalg.norm(mag)
        if not os.path.exists("/home/mord/tensor/deployed_tensor_model/last.ckpt"):
            pass
        else:
            self.model = LitModel()
            self.model.load_from_checkpoint("/home/mord/tensor/deployed_tensor_model/last.ckpt")
            self.model.freeze()
            custom_reward.k = 1
        direction = [mag[i]/self.old_mag for i in range(0, 11)]
        self.old_dir = direction
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = 5000.0

    def execution(self, data):
        if custom_reward.k == 0:
            return 0.0
        else:
            data1 = self.prep(data=data)
            m = self.model([torch.tensor(data1[0], dtype= torch.float32), torch.tensor(data1[1], dtype= torch.float32)])
            v = 2*float(m) -1
            return v


    def get_data(old_state, new_state, reward):
        df.loc[len(df)] = [old_state, new_state, reward]
        if len(df)==4096:
            df2 = pd.DataFrame(columns=['old_state', 'new_state', 'reward'])
            df2 = df.copy()
            df2.to_json(f"data/dataset{custom_reward.i}.json", compression = "infer", orient = "split", index= False)
            df.drop(df.index, inplace=True)
            custom_reward.i = custom_reward.i + 4096
        else:
            pass


    def reward(self, reward):
        c_state = custom_obs.current_state
        rc_state = [c_state[i] for i in iter(custom_reward.obs_state)]
        nmag = [self.final_state[i] - rc_state[i] for i in range(0, 11)]
        self.new_mag = np.linalg.norm(nmag)
        newdir = [nmag[i]/self.new_mag for i in range(0, 11)]
        self.new_dir = newdir
        new_theta = np.arccos(np.dot(self.new_dir, self.old_dir))
        delta = np.abs(new_theta - self.old_theta)
        consistency = 0
        if self.new_mag < self.old_mag:
            if consistency < 0:
                consistency = 0
            reward1 = 10000*delta + 0.1*(consistency)**(consistency)
            consistency += 2

        elif self.new_mag > self.old_mag:
            if consistency > 0:
                consistency = 0
            reward1 = -(10000*delta + 0.1*(consistency)**(consistency))
            consistency += 2
        v = custom_reward.execution(self= custom_reward ,data=[self.o_state[0:27], c_state[0:27]])
        total_reward = reward + v
        custom_reward.get_data(self.o_state[0:27], c_state[0:27], reward=total_reward)
        self.o_state = c_state
        self.old_dir = self.new_dir
        self.old_mag = self.new_mag
        return total_reward
