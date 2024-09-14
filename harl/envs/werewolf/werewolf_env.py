import copy
import numpy as np
from gym.spaces import Discrete, Box
from MafiaGPT_RL.core.game_env import WerewolfGameEnv

env_num_agents = {
    "basic_scenario" : 7
}


class WerewolfEnv:
    def __init__(self, scenario, player_config_path, init_id = 1, openai_client = None, data_path = None):
        self.id = init_id
        self.scenario = scenario
        self.player_config_path = player_config_path
        self.env = WerewolfGameEnv(id = self.id, train = True, openai_client = openai_client, data_path = data_path)
        self.openai_client = openai_client
        self.data_path = data_path
        self.env.set_players(player_config_path)
        self.n_agents = env_num_agents[scenario] if scenario in env_num_agents else env_num_agents["basic_scenario"]
        self.share_observation_space = self.env.shared_observation_space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def reset(self):
        self.id += 1
        self.env = WerewolfGameEnv(id = self.id, train = True, openai_client = self.openai_client, data_path = self.data_path)
        self.env.set_players(self.player_config_path)
    
    def step(self, actions):
        return self.env.step(actions)
    
    def seed(self, seed):
        self.env.seed(seed)
        return seed

    def render(self):
        pass
    
    def close(self):
        self.env.close()
        return