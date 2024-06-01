from gymnasium.envs.registration import register

register(
    id="custom-environments/TrucoUruguayo-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
)
