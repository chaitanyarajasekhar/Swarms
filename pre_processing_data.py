import numpy as np
import matplotlib.pyplot as plt
import time

import argparse
import os



def main():

    positions = np.load(os.path.join(ARGS.data_dir, 'test_position.npy')) #"chaser_20_not_random/test_random_position.npy")
    velocities = np.load(os.path.join(ARGS.data_dir, 'test_velocity.npy'))

    if ARGS.save_file:
        temp_new_pos = np.squeeze(positions[:,:,0,:])
        temp_new_vel = np.squeeze(velocities[:,:,0,:])

        pos_vel_file = np.concatenate([temp_new_pos, temp_new_vel], axis = 1)


        for i in range(positions.shape[2] - 1):
            temp_new_pos = np.squeeze(positions[:,:,i+1,:])
            temp_new_vel = np.squeeze(velocities[:,:,i+1,:])
            pos_vel_file = np.concatenate([pos_vel_file, temp_new_pos, temp_new_vel], axis = 1)


        np.savetxt(os.path.join(ARGS.data_dir, 'chaser_{}.csv'.format(
            ARGS.save_prefix)), pos_vel_file)#, fmt = '%1.6f')


    if ARGS.show_simulation:

        for i in range(positions.shape[1]):

            plt.clf()
            plt.axis([-1.6, 1.6, -1, 1])

            plt.scatter(np.squeeze(positions[:,i,:,0]),np.squeeze(positions[:,i,:,1]))#


            plt.pause(ARGS.dt)

            
        plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        help='data directory')
    parser.add_argument('--dt', type=float, default = 0,
                        help='time step')
    parser.add_argument('--show-simulation', action = 'store_true', default = False,
                        help='data directory')
    parser.add_argument('--save-file', action = 'store_true', default = False,
                        help='data directory')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='save file prefix')

    ARGS = parser.parse_args()


    main()
