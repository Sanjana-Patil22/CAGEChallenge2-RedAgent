
import inspect
from CybORG import CybORG

from .MainAgent import MainAgent
from CybORG.Agents import B_lineAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

class WrappedMainAgent(MainAgent):
    
    def __init__(self,lr=0.002, betas=[0.9, 0.990], gamma=0.99, K_epochs=4, eps_clip=0.2, restore=False, ckpt=None,
                 deterministic=False, training=True,):
        super().__init__()
        scenario = 'Scenario2'
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'
        ori_cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
        self.w_env = ChallengeWrapper2(env=ori_cyborg, agent_name='Blue')
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.restore = restore
        self.ckpt = ckpt
        self.deterministic = deterministic
        self.training = training
        self.action_space = [133, 134, 135, 139, 3, 4, 5, 9, 16, 17, 18, 22, 11, 12, 13, 14, 141, 142, 143, 144,
                             132, 2, 15, 24, 25, 26, 27]

    def get_action(self, observation, action_space=None):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        conv_obs = self.w_env.env.env.env.observation_change(observation)
        action = super().get_action(conv_obs)
        conv_action = self.w_env.env.env.possible_actions[action]
        return conv_action
    
    def set_initial_values(self, action_space, observation=None):
        # conv_action_space = self.action_space
        conv_action_space = [133, 134, 135, 139, 3, 4, 5, 9, 16, 17, 18, 22, 11, 12, 13, 14, 141, 142, 143, 144,
                             132, 2, 15, 24, 25, 26, 27]
        self.input_dims = 52
        super().set_initial_values(conv_action_space, observation)

    def train(self, arg):
        pass