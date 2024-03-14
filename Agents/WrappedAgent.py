
import inspect
import numpy as np

from CybORG import CybORG
from CybORG.Agents import BaseAgent
from CybORG.Shared import Results

from CybORG.Agents import B_lineAgent

from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

from Agents.MainAgent import MainAgent

class WrappedBlueAgent(BaseAgent):
    def __init__(self, agent=None):
        scenario = 'Scenario2'
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'
        red_agent = B_lineAgent
        ori_cyborg = CybORG(path, 'sim')
        agent_name = 'Blue'
        self.w_env = ChallengeWrapper2(env=ori_cyborg, agent_name=agent_name)
        if agent is not None:
            self.agent = agent
        else:
            self.agent = MainAgent()

    def train(self, results: Results):
        pass
    def reset(self):
        self.agent = MainAgent()

    def get_action(self, observation, action_space):
        conv_obs = self.w_env.env.env.env.observation_change(observation)
        action = self.agent.get_action(conv_obs, action_space)
        conv_action = self.w_env.env.env.possible_actions[action]
        return conv_action

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass


class WrappedRedAgent(BaseAgent):
    def __init__(self, agent=None):
        scenario = 'Scenario2'
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'
        red_agent = B_lineAgent
        ori_cyborg = CybORG(path, 'sim')
        agent_name = 'Red'
        self.w_env = ChallengeWrapper2(env=ori_cyborg, agent_name=agent_name)
        if agent is not None:
            self.agent = agent
        else:
            raise NotImplementedError

    def train(self, results: Results):
        pass

    def get_action(self, observation, action_space):
        conv_obs = self.w_env.env.env.env.observation_change(observation)
        action = self.agent.get_action(conv_obs, action_space)
        conv_action = self.w_env.env.env.possible_actions[action]
        return conv_action

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass