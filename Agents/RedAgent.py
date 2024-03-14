from CybORG.Agents import BaseAgent
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import random

class LoggingCallback(BaseCallback):
    """
    Custom logging callback to provide more information about the agent actions being taken and the associated rewards.
    pring_freq: This parameter is used to specify how often the information is printed. By default this is every 10 timesteps.
    verbose: If this is greater than 0, logging is enabled.
    """
    def __init__(self, verbose=0, print_freq=10):
        super(LoggingCallback, self).__init__(verbose)

        self.num_timesteps = 0
        self.verbose = verbose
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        infos = self.locals['infos']
        self.num_timesteps += 1
        if self.verbose and self.num_timesteps % self.print_freq == 0:
            for info in infos:
                print(f"Reward: {info.get('reward')}, Action: {info.get('action')}")

        return True

class RedAgent(BaseAgent):
    def __init__(self, env, model_file: str = None) -> None:
        super().__init__()
        if model_file is not None:
            self.model = PPO.load(model_file)
            return

        self.model = PPO('MlpPolicy', env, verbose=1, n_steps=30, tensorboard_log="logs")
        self.callback = LoggingCallback(verbose=1)
        #self.callback = CheckpointCallback(
        #    save_freq=500,
        #    save_path="models/Scenario2/checkpoints/",
        #    name_prefix="red_ppo_agent",
        #    save_replay_buffer=True,
        #    save_vecnormalize=True
        #)

    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        action, _states = self.model.predict(observation)
        return action

    def train(self, timesteps):
        """allows an agent to learn a policy"""
        self.model.learn(total_timesteps=timesteps, callback=self.callback)


