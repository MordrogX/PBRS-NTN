import gym
from stable_baselines3 import PPO
import os
from reward import custom_reward, custom_obs



if not os.path.exists("model/tensor"):
    os.makedirs("model/tensor")


if not os.path.exists("data"):
    os.makedirs("data")

if not os.path.exists("logs"):
    os.makedirs("logs")

env = gym.make("Ant-v2")

env = custom_obs(env)


env = custom_reward(env, -1000, 4000)


'''

class custom_reward(gym.RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.total_reward = None
        self.env = env
        initial_state = self.env.reset()
        self.old_displ = np.linalg.norm([initial_state[0], initial_state[1]])
        self.old_vel = np.linalg.norm([initial_state[2], initial_state[3]])

        self.vel_angle = np.arccos(np.dot([initial_state[0], initial_state[1]], [initial_state[2], initial_state[3]]))

        self.min_reward = -1000.0
        self.max_reward = 1000.0
        self.reward_range(2000.0)

    def reward(self, reward):
        def pos_reward():
            self.new_displ = np.linalg.norm([self.env.observation_space[0], self.env.observation_space[1]])
            delta = 0

            if self.new_displ < self.old_displ:
                if delta < 0:
                    delta = 0
                reward_pos = 10 + (delta)**2
                delta += 2


            elif self.new_displ > self.old_displ:
                if delta > 0:
                    delta = 0
                reward_pos = -(10 + (delta)**2)
                delta += 2


            self.old_displ = self.new_displ

            return reward_pos

        
        def vel_reward():
            self.new_vel_angle = np.arccos(np.dot([self.env.observation_space[0], self.env.observation_space[1]], [self.env.observation_space[2], self.env.observation_space[3]]))
            ang_delta = 0

            if self.new_vel_angle < self.vel_angle:
                if ang_delta < 0:
                    ang_delta = 0
                reward_vel = 10 * (ang_delta)**2
                ang_delta += 2


            elif self.new_vel_angle > self.vel_angle:
                if ang_delta > 0:
                    ang_delta = 0
                reward_vel = -10 * (ang_delta)**2 
                ang_delta += 2

            self.vel_angle = self.new_vel_angle

            return reward_vel

        def ang_reward():
            self.angle = self.env.observation_space[4]
            if self.angle >= 0:
                self.angle = -(10*self.angle) + 15
                reward_ang = -(self.angle)**2
        
            if self.angle < 0:
                self.angle = 10*(self.angle) + 15
                reward_ang =  -(self.angle)**2
        
            return reward_ang
        
        def ground_reward():
            if self.env.observation_space[6] == 1 and self.env.observation_space[7] == 1:
                reward_touch = 50 - 10(self.env.observation_space[0])**2
            elif self.env.observation_space[6] == 1:
                reward_touch = 20 - 10(self.env.observation_space[0])**2
            elif self.env.observation_space[7] == 1:
                reward_touch = 20 - 10(self.env.observation_space[0])**2
            return reward_touch
        r_list = [[pos_reward(), 1], [vel_reward(), 2], [ang_reward(), 3], [ground_reward(), 4]]
        r_list.sort(reverse=True)
        r_change = [r_list[0][0] - r_list[3][0], r_list[1][0] - r_list[3][0], r_list[2][0] - r_list[3][0], r_list[3][0] - r_list[3][0]]
        r_total = pos_reward() + vel_reward() + ang_reward() + ground_reward()
        ratio  = [1- r_change[0]/r_total, 1- r_change[1]/r_total, 1- r_change[2]/r_total, 1- r_change[3]/r_total]

        for i in range(3):
            if r_list[i][1] == 1:
                w1 = ratio[i]

            elif r_list[i][1] == 2:
                w2 = ratio[i]

            elif r_list[i][1] == 3:
                w3 = ratio[i]

            elif r_list[i][1] == 4:
                w4 = ratio[i]

        self.total_reward = w1*pos_reward() + w2*vel_reward() + w3*ang_reward() + w4*ground_reward()
        return self.total_reward
'''



try:
    model = PPO.load("model/tensor.zip", env=env)
    print(".....Loading Previous Model.....")


except:
    print(".....Creating New Model.....")
    model = PPO('MlpPolicy', env, verbose=1 , tensorboard_log="logs", learning_rate=3e-4, ent_coef=0.0, batch_size=32, gae_lambda=0.95, gamma=0.99, clip_range=0.1, device='cpu')


try:
    while True:
        model.learn(total_timesteps=2048*1400,reset_num_timesteps= False, tb_log_name="TENSOR")
        model.save(f"model/tensor/{2048*350}")

except KeyboardInterrupt:
    print("Exiting training")
    print("Saving model")
    model.save("model/tensor")
    print("Save complete")


