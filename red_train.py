# Implementation of the red evaluation script
# This script is used to evaluate the performance of a red agent against a blue agent
# The blue agent is https://github.com/john-cardiff/-cyborg-cage-2.git
# Modified by Prof. H. Sasahara
import inspect
import random
import time

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.MainAgent import MainAgent
from Agents.WrappedAgent import WrappedBlueAgent
from Agents.RedAgent import RedAgent

from stable_baselines3.common.vec_env import SubprocVecEnv

MAX_EPS = 100
NUM_CPU = 16
agent_name = 'Red'
random.seed(153)


def make_env(scenario: str, rank: int, seed: int = 0):
    """
    Function to make a new Cyborg environment given some scenario. The main reason for this function is to allow for the creation of multiple environments to be used with the SubprocVecEnv.
    """

    # Define a new function that will be called to create the new CybORG environment.
    def _init():
        # Get path to scenario file.
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

        # Create default unwrapped cyborg gym with the WrappedBlueAgent
        # as the adversary for this game.
        cyborg = CybORG(path, 'sim', agents={'Blue': WrappedBlueAgent})

        # Instantiate the wrapper around the newly created cyborg gym.
        env = ChallengeWrapper2(env=cyborg, agent_name="Red", max_steps=30)

        return env

    return _init

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'

    # Set up environment with blue agent running in the background and
    # red agent as the main agent.
    vec_env = SubprocVecEnv([make_env(scenario, i) for i in range(NUM_CPU)])

    # Create a new red agent passing the vectorized environment for faster training times.
    red_agent = RedAgent(vec_env)

    # Train the red agent for 50k timesteps.
    red_agent.train(100000)

    # Save the final agent to a zip file by the name of red_ppo_agent.zip
    # This zip file contains all the .pth policy files within.
    red_agent.model.save("models/Scenario2/red_ppo_agent")
