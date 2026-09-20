from gymnasium.envs.registration import register

from truco_uruguayo_gymnasium.envs.truco_uruguayo_env import TrucoUruguayoEnv

register(
    id="TrucoUruguayo-v0",
    entry_point="truco_uruguayo_gymnasium:TrucoUruguayoEnv",
)

# https://github.com/markub3327/flappy-bird-gymnasium/tree/main
