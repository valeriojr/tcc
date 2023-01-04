import argparse

import glfw
from matplotlib import pyplot
import numpy
import tensorflow as tf

import rendering
from agent import Agent
from environment import Environment


def run_episode(agent, environment, max_length, training=True, time_step=32):
    state = environment.reset()
    #
    # for i in range(environment.agents):
    #     print(state[i])
    #     cv2.imwrite(f'res/state_{i}.png', state[i] * 255)

    accumulated_return = numpy.zeros(environment.agents)

    step = 0
    while True:
        if training:
            final_step = True if step == max_length - 1 else False
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                action = agent.act(state)
                value = agent.critic(state.copy())
                done, r, next_state = environment.step(agent.sample_actions(action), time_step=time_step,
                                                       final_step=final_step)

                accumulated_return += r
                critic_loss = tf.reduce_sum(agent.criticize(value, r, next_state)) / environment.agents
                actor_loss = agent.get_actor_loss(action, critic_loss)

            agent.update(actor_tape, actor_loss, critic_tape, critic_loss)

            step += 1
            if step >= max_length:
                break
        else:
            action = agent.act(state)
            done, r, next_state = environment.step(agent.sample_actions(action), time_step=time_step)
            if done.all():
                break

        glfw.poll_events()
        if glfw.window_should_close(environment.renderer.window):
            exit(-1)

    return accumulated_return


def main(agents, max_episodes, max_episode_length, time_step, training=True):
    state_renderer = rendering.PyOpenGLRenderer(96 * 10, 96 * 5, 45)

    agent = Agent(state_shape=(96, 96, 3), actions=2, learning_rate=0.001, discount_factor=0.9, saved_actor='actor.h5', saved_critic='critic.h5')
    environment = Environment(agents, state_renderer)

    return_per_episode = numpy.zeros((max_episodes, agents))

    for episode in range(max_episodes):
        print(f'Episode #{episode: 4d}')
        return_per_episode[episode] = run_episode(agent, environment, max_episode_length, training=training,
                                                  time_step=time_step)

        agent.actor.model.save('actor.h5')
        agent.critic.model.save('critic.h5')

        pyplot.clf()
        pyplot.fill_between(numpy.arange(episode + 1), return_per_episode[:episode + 1].min(axis=1),
                            return_per_episode[:episode + 1].max(axis=1), alpha=0.5)
        pyplot.plot(numpy.arange(episode + 1), return_per_episode[:episode + 1].mean(axis=1), label='Average return')
        pyplot.title(f'{environment.agents} agents')
        pyplot.xlabel('Episode')
        pyplot.ylabel('Return')
        pyplot.show(block=False)
        pyplot.savefig('average_return_per_episode')
        pyplot.pause(0.00001)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('train', 'test'))
    parser.add_argument('--agents', type=int, help='Number of agents to simulate')
    parser.add_argument('--environment', type=str)
    parser.add_argument('--max_episodes', type=int, help='Number of episodes to run')
    parser.add_argument('--max_episode_length', type=int, help='Maximum number of steps to simulate in each episode')
    parser.add_argument('--timestep', type=int, help='Simulation timestep in ms')

    args = parser.parse_args()
    training = args.command == 'train'
    main(args.agents, args.max_episodes, args.max_episode_length, args.timestep, training)
