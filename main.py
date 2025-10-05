from src.swarm_env import SwarmEnv

ITERATION_NUMBER = 1000
WORLD_SIZE = 20
AGENTS_NUMBER = 5

def main():
    env = SwarmEnv(n_agents=AGENTS_NUMBER, world_size=WORLD_SIZE)
    env.reset()
    for _ in range(ITERATION_NUMBER):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        print(f"Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            env.reset()
    env.close()

if __name__ == "__main__":
    main()