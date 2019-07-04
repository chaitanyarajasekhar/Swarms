import numpy as np
import matplotlib.pyplot as plt
import time

import argparse
import os

def main():

    pose = np.squeeze(np.load(os.path.join(ARGS.data_dir, ARGS.prefix+'_position.npy')))

    if ARGS.show_simulation:

        for i in range(pose.shape[0]):

            plt.clf()
            plt.axis([-1.6, 1.6, -1, 1])

            plt.scatter(np.squeeze(pose[i,:,0]),np.squeeze(pose[i,:,1]))

            plt.pause(0.1)

        plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        help='data directory')
    parser.add_argument('--show-simulation', action = 'store_true', default = False,
                        help='show simulation')
    parser.add_argument('--prefix', type = str, default = None,
                        help = 'prefix of the file')

    ARGS = parser.parse_args()


    main()
