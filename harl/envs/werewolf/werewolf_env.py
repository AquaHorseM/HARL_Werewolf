import copy, json
import numpy as np
from gym.spaces import Discrete, Box
from MafiaGPT_RL.core.game_env import WerewolfGameEnv

env_num_agents = {
    "basic_scenario" : 7
}


class WerewolfEnv:
    def __init__(self, args):
        self.id = args.get("init_id", 0)
        self.scenario = args.get("scenario", "basic_scenario")
        self.player_config_path = args.get("player_config_path", "MafiaGPT_RL/core/player_config.yaml")
        player_configs = json.load(open(self.player_config_path))["players"]
        self.openai_client = args.get("openai_client", None)
        self.data_path = args.get("data_path", None)
        self.env = WerewolfGameEnv(id = self.id, train = True, openai_client = self.openai_client, data_path = self.data_path)
        
        self.env.set_players(player_configs)
        self.env.init_env()
        self.n_agents = env_num_agents[self.scenario] if self.scenario in env_num_agents else env_num_agents["basic_scenario"]
        self.share_observation_space = self.env.shared_observation_space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def reset(self):
        self.id += 1
        self.env = WerewolfGameEnv(id = self.id, train = True, openai_client = self.openai_client, data_path = self.data_path)
        self.env.set_players(json.load(open(self.player_config_path))["players"])
    
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