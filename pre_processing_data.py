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

        print(positions[:,:,0,0].shape)
        #print(pos_vel_file)

        np.savetxt(os.path.join(ARGS.data_dir, 'chaser_{}.txt'.format(
            ARGS.save_prefix)), pos_vel_file, delimiter = ',', fmt = '%1.6f')
    # plt.hold(True)

    # plt.ion()

    if ARGS.show_simulation:

        for i in range(positions.shape[1]):

            # plt.plot(np.squeeze(positions[:,i,0,0]),np.squeeze(positions[:,i,0,1]),'o')
            # plt.plot(np.squeeze(positions[:,i,1,0]),np.squeeze(positions[:,i,1,1]),'s')
            # plt.plot(np.squeeze(positions[:,i,2,0]),np.squeeze(positions[:,i,2,1]),'*')
            # plt.plot(np.squeeze(positions[:,i,3,0]),np.squeeze(positions[:,i,3,1]),'D')
            # plt.plot(np.squeeze(positions[:,i,4,0]),np.squeeze(positions[:,i,4,1]),'X')
            plt.clf()
            plt.axis([-1.6, 1.6, -1, 1])

            plt.scatter(np.squeeze(positions[:,i,:,0]),np.squeeze(positions[:,i,:,1]))#

            # for j in range(20):
            #
            #
            #     plt.plot(np.squeeze(positions[:,i,j,0]),np.squeeze(positions[:,i,j,1]))#, marker = 'o', c = 'b')
            # plt.plot(np.squeeze(positions[:,i,1,0]),np.squeeze(positions[:,i,1,1]), marker = 's', c = 'r')
            # plt.plot(np.squeeze(positions[:,i,2,0]),np.squeeze(positions[:,i,2,1]), marker = '*', c = 'g')
            # plt.plot(np.squeeze(positions[:,i,3,0]),np.squeeze(positions[:,i,3,1]), marker = 'D', c = 'm')
            # plt.plot(np.squeeze(positions[:,i,4,0]),np.squeeze(positions[:,i,4,1]), marker = 'X', c = 'k')

            plt.pause(0.01)

            # plt.draw()
            # time.sleep(0.1)

            # plt.hold(True)

        plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        help='data directory')
    # parser.add_argument('--file-prefix', type=str,
    #                     help='file prefix')
    parser.add_argument('--show-simulation', action = 'store_true', default = False,
                        help='data directory')
    parser.add_argument('--save-file', action = 'store_true', default = False,
                        help='data directory')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='save file prefix')

    ARGS = parser.parse_args()


    main()
