import os
import time
import argparse
import numpy as np

import matplotlib.pyplot as plt
from classes import ParticleChaser
import utils

class clickGoal:
    def __init__(self,fig):
        self.goal_position = np.zeros(2)
        self.goal_velocity = np.zeros(2)
        self.goal_acceleration = np.zeros(2)
        self.cid = fig.canvas.mpl_connect('button_press_event', self)
        self.particles, self.edges = create_chasers(n = ARGS.num_particles, m = ARGS.num_targets, radius = ARGS.radius,
                                        max_speed = ARGS.max_speed, max_acceleration = ARGS.max_acc,
                                        initial_vel = ARGS.initial_vel_mag, goal_position = None,
                                        not_random_target = ARGS.not_random_target, circular_init = ARGS.circular_init)
        self.position_data = []
        self.velocity_data = []
        self.time_step = 0

    def __call__(self,event):
        print('time step =', self.time_step)
        self.goal_position[0] = event.xdata
        self.goal_position[1] = event.ydata
        self.time_step += 1
        self.update_step()

    def simulation_pos_vel(self):
        return self.position_data, self.velocity_data, self.edges

    def update_step(self):
        step_position = []
        step_velocity = []
        self.particles[0].reset(self.goal_position, self.goal_velocity, self.goal_acceleration)

        for p in self.particles:
            step_position.append(p.position.copy())
            step_velocity.append(p.velocity.copy())
            p.decide()

        for p in self.particles:
            p.move(ARGS.dt)

        self.position_data.append(step_position)
        self.velocity_data.append(step_velocity)


def create_chasers(n,m, radius = 0.8, max_speed = None, max_acceleration = None,
                    initial_vel = None, goal_position = None, not_random_target = False, circular_init = False):
    """
    Create n particle chasers.
    Each particle chases the previous one in the list of particles.
    """
    if n < 1:
        raise ValueError('n must be a positive integer')

    if m < 1 or m > n - 1:
        raise ValueError('m must be a positive integer less than n')

    particles = []
    if goal_position is None:
        goal_position = np.random.uniform(-0.9,0.9,2)

    particles.append(ParticleChaser(goal_position, np.zeros(2), ndim=2, max_speed=0,
                    max_acceleration=0))

    for i in range(n):
        r = radius

        if circular_init is False:
            theta = np.random.rand() * 2 * np.pi
        else:
            theta = i * 2* np.pi/ n

        x, y = r * np.cos(theta), r * np.sin(theta)
        if initial_vel is not None:
            v = np.random.uniform(-initial_vel,initial_vel,2)
        else:
            v = np.random.uniform(-max_speed, max_speed, 2)

        particles.append(ParticleChaser((x, y), v, ndim=2, max_speed=max_speed,
                        max_acceleration=max_acceleration))

    if not_random_target is True:
        for i in range(1,n+1):
            particles[i].add_target(particles[(i+1)%n])
        edges = chasers_edges(n+1)
        return particles, edges
    else:
        edges = np.zeros((n+1, n+1))
        particle_idxs = np.arange(n+1)
        for i, p in enumerate(particles):
            if i != 0:
                for j in np.random.choice(particle_idxs[particle_idxs != i], m, replace=False):
                    edges[j, i] = 1  # j is i's target, thus j influences i through edge j->i.
                    p.add_target(particles[j])

        return particles, edges


def chasers_edges(n):
    """
    Edges for a list of chaser particles in which each agent chases its predecessor in the list.
    A 1 at Row i, Column j means Particle i influences Particle j. No influence
    is represented by 0.
    """
    matrix = np.zeros((n, n), dtype=int)
    for i in range(1,n):
        matrix[(i+1) % n, i] = 1

    return matrix

def simulationClick():

    np.random.seed()
    fig, ax = plt.subplots()
    ax.set_xlim([-1.6,1.6])
    ax.set_ylim([-1,1])
    clickGoalObject = clickGoal(fig)
    plt.show()
    position_data, velocity_data, edges = clickGoalObject.simulation_pos_vel()

    return position_data, velocity_data, edges

def simulation(_):

    np.random.seed()

    particles, edges = create_chasers(n = ARGS.num_particles, m = ARGS.num_targets, radius = ARGS.radius,
                    max_speed = ARGS.max_speed, max_acceleration = ARGS.max_acc,
                    initial_vel = ARGS.initial_vel_mag, goal_position = None,
                    not_random_target = ARGS.not_random_target, circular_init = ARGS.circular_init)

    position_data = []
    velocity_data = []

    for _ in range(ARGS.steps):
        step_position = []
        step_velocity = []
        for p in particles:
            step_position.append(p.position.copy())
            step_velocity.append(p.velocity.copy())

            p.decide()

        for p in particles:
            p.move(ARGS.dt)

        particles[0].reset(np.random.uniform(-0.9,0.9, 2))

        position_data.append(step_position)
        velocity_data.append(step_velocity)

    return position_data, velocity_data, edges


def main():
    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)

    if ARGS.click_goal is True:
        position_data_all, velocity_data_all, edge_data_all = simulationClick()
    else:
        position_data_all, velocity_data_all, edge_data_all = utils.run_simulation(simulation,
                                                                                ARGS.instances,
                                                                                ARGS.processes,
                                                                                ARGS.batch_size)

    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_position.npy'), position_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_velocity.npy'), velocity_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_edge.npy'), edge_data_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-particles', '-n', type=int, default=5,
                        help='number of particles')
    parser.add_argument('--num-targets', '-m', type=int, default=1,
                        help='number of targets for each particle')
    parser.add_argument('--instances', type=int, default=1,
                        help='number of instances to run')
    parser.add_argument('--steps', type=int, default=50,
                        help='number of time steps per simulation')
    parser.add_argument('--dt', type=float, default=0.3,
                        help='unit time step')
    parser.add_argument('--save-dir', type=str,
                        help='name of the save directory')
    parser.add_argument('--prefix', type=str, default='',
                        help='prefix for save files')
    parser.add_argument('--save-edges', action='store_true', default=False,
                        help='Deprecated. Now edges are always saved.')
    parser.add_argument('--processes', type=int, default=1,
                        help='number of parallel processes')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='number of simulation instances for each process')
    parser.add_argument('--max-speed', type=float, default=0.4,
                        help='maximum velocity magnitude for agents')
    parser.add_argument('--max-acc', type=float, default=3,
                        help='maximum acceleration magnitude for agents')
    parser.add_argument('--radius', type=float, default=0.9,
                        help='number of simulation instances for each process')
    parser.add_argument('--initial-vel-mag', type=float, default=None,
                        help='initial velocity magnitude')
    parser.add_argument('--circular-init', action='store_true', default=False,
                        help='initialize agnets in circular formation')
    parser.add_argument('--not-random-target', action = 'store_true', default = False,
                        help = 'circular target initialization')
    parser.add_argument('--click-goal', action = 'store_true', default = False,
                        help = 'click based goal')

    ARGS = parser.parse_args()

    ARGS.save_dir = os.path.expanduser(ARGS.save_dir)

    main()
