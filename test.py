import gym
import numpy as np
from stable_baselines3 import  PPO
import pandas as pd
import os


env = gym.make("Ant-v2")
env.render(mode='human')


model = PPO.load("model/tensor.zip", env=env)



while True:
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
