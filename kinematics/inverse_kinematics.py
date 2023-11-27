'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''

from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity
from math import atan2, asin, acos, pow, sqrt, cos, sin, pi, atan
import numpy as np


class InverseKinematicsAgent(ForwardKinematicsAgent):
    # def inverse_kinematics(self, effector_name, transform):
    #     '''solve the inverse kinematics
    #
    #     :param str effector_name: name of end effector, e.g. LLeg, RLeg
    #     :param transform: 4x4 transform matrix
    #     :return: list of joint angles
    #     '''
    #     joint_angles = {}
    #     # YOUR CODE HERE
    #     joint_angles = {key: self.perception.joint[key] for key in self.chains[effector_name]}
    #     target_angles = self.extract_transform_components(transform, d=0)
    #
    #     lambda_ = 0.001
    #     for i in range(3000):
    #         self.forward_kinematics(joint_angles)
    #
    #         T_matrices = [self.transforms[joint] for joint in self.chains[effector_name]]
    #         Te = self.extract_transform_components(T_matrices[-1], 1)
    #
    #         e = target_angles - Te
    #
    #         T_components = np.array([self.extract_transform_components(transform, 1) for transform in T_matrices])
    #
    #         J = Te - T_components
    #         J[-1, :] = 1
    #
    #         d_theta = lambda_ * np.dot(np.dot(J.T, np.linalg.pinv(np.dot(J, J.T))), e)
    #
    #         for i, name in enumerate(self.chains[effector_name]):
    #             joint_angles[name] += d_theta[i]
    #
    #         if np.linalg.norm(d_theta) < 1e-4:
    #             break
    #
    #     return joint_angles
    def decode_transform_matrix(self, T, d=1):
        # Decode offsets
        x, y, z = 0, 0, 0
        if d == 0:
            # if offsets are in the last row
            x, y, z = T[3, 0], T[3, 1], T[3, 2]
        elif d == 1:
            # if offsets are in the last column
            x, y, z = T[0, 3], T[1, 3], T[2, 3]

        # Decode angles.
        angle_x, angle_y, angle_z = 0, 0, 0
        # if T[0, 0] == 1:
        #     angle_x = atan2(T[2, 1], T[1, 1])
        # elif T[1, 1] == 1:
        #     angle_y = atan2(T[0, 2], T[0, 0])
        # elif T[2, 2] == 1:
        #     angle_z = atan2(T[1, 0], T[0, 0])

        angle_x = -asin(T[1, 2] / sqrt(1 - pow(T[0, 2], 2)))
        angle_y = asin(T[0, 2])
        angle_z = asin(T[0, 1] / sqrt(1 - pow(T[0, 2], 2)))

        components = [x, y, z, angle_x, angle_y, angle_z]
        return np.array(components)
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        # Set actual angles and target angles
        actual_angles = {key: self.perception.joint[key] for key in self.chains[effector_name]}
        target_angles = np.array(self.decode_transform_matrix(transform)).T

        theta = np.random.random(len(self.chains[effector_name])) * 1e-5
        lambda_ = 1
        max_step = 0.1
        for i in range(3000):
            self.forward_kinematics(actual_angles)

            T_matrices = [0] * len(self.chains[effector_name])
            for i, name in enumerate(self.chains[effector_name]):
                T_matrices[i] = self.transforms[name]

            Te = np.array([self.decode_transform_matrix(T_matrices[-1])])
            e = target_angles - Te
            T_matrices = np.array([self.decode_transform_matrix(i) for i in T_matrices[0:len(self.chains[effector_name])]])
            J = (Te - T_matrices).T
            J[-1, :] = 1
            d_theta = lambda_ * np.dot(np.dot(J.T, np.linalg.pinv(np.dot(J, J.T))), e.T)

            for i, name in enumerate(self.chains[effector_name]):
                actual_angles[name] += np.asarray(d_theta.T)[0][i]

            if np.linalg.norm(d_theta) < 1e-4:
                break

        return actual_angles

        #
        # lambda_ = 0.001
        # epsilon = 1e-6  # Small value for numerical differentiation
        # max_step = 0.1  # Maximum allowable change in the error vector
        #
        # for i in range(1000):
        #     self.forward_kinematics(actual_angles)
        #
        #     T_matrices = [self.transforms[joint] for joint in self.chains[effector_name]]
        #     Te = self.extract_transform_components(T_matrices[-1], 1)
        #
        #     e = target_angles - Te
        #     e = 0
        #
        #     Clamp the error
        # e = np.clip(e, -max_step, max_step)
        #
        # J = np.zeros((6, len(actual_angles)))  # Initialize the Jacobian matrix

        # for idx, joint in enumerate(self.chains[effector_name]):
        #     Save the original angle
        #     original_angle = actual_angles[joint]
        #
        # Perturb the joint angle by epsilon
        # actual_angles[joint] += epsilon
        # self.forward_kinematics(actual_angles)
        #
        # Compute the end effector position with the perturbed joint angle
        # Te_perturbed = self.extract_transform_components(self.transforms[self.chains[effector_name][-1]], 1)
        #
        # Restore the original joint angle
        # actual_angles[joint] = original_angle
        #
        # The difference in end effector position divided by epsilon is the partial derivative
        # J[:, idx] = (Te_perturbed - Te) / epsilon
        #
        # Calculate the pseudo-inverse of the Jacobian
        # J_pinv = np.linalg.pinv(J)
        #
        # Calculate the change in joint angles
        # d_theta = lambda_ * np.dot(J_pinv, e)
        #
        # Update the joint angles
        # for i, joint in enumerate(self.chains[effector_name]):
        #     actual_angles[joint] += d_theta[i]
        #
        # Check for convergence
        # if np.linalg.norm(d_theta) < 1e-4:
        #     break
        #
        # return actual_angles

    # def TU_Crete():
    #     # A0 = self.chains[effector_name[]]
    #     # YOUR CODE HERE
    #     # joint_angles = {}
    #     #
    #     # for chain in self.chains:
    #     #     for joint in self.chains[chain]:
    #     #         joint_angles[joint] = self.perception.joint[joint]
    #     # self.forward_kinematics(joint_angles)
    #
    #     if effector_name == 'LLeg':
    #         start_effector = self.chains[effector_name][0]
    #         # end_efector = self.chains[effector_name][-1]
    #         A_base_0 = self.transforms[start_effector]
    #         A_base_0_inv = np.linalg.pinv(A_base_0)
    #         # A_6_end = self.transforms[end_efector]
    #         # A_end_inv = np.linalg.pinv(A_6_end)
    #         # T_i = np.dot(A_base_0_inv, transform)
    #
    #         # T_i = np.dot(T_i, )
    #         R_x = np.array([[1, 0, 0, 0],
    #                         [0, cos(pi / 4), -sin(pi / 4), 0],
    #                         [0, sin(pi / 4), cos(pi / 4), 0],
    #                         [0, 0, 0, 1]])
    #         R_y = np.array([[cos(-pi / 2), 0, sin(-pi / 2), 0],
    #                         [0, 1, 0, 0],
    #                         [-sin(-pi / 2), 0, cos(-pi / 2), 0],
    #                         [0, 0, 0, 1]])
    #         R_z = np.array([[cos(pi), 0, sin(pi), 0],
    #                         [0, 1, 0, 0],
    #                         [-sin(pi), 0, cos(pi), 0],
    #                         [0, 0, 0, 1]])
    #
    #         T_i = np.linalg.pinv(
    #             R_x * A_base_0_inv * transform)  # Esta raro que en el paper R es 3x3 y aqui sea 4*4
    #
    #         # l1 = (self.joint_lengths['LKneePitch'])
    #         l1 = 100 / 1000
    #         l2 = 102.9 / 1000
    #         distance = np.linalg.norm([transform[0, -1], transform[1, -1], transform[2, -1]])
    #         theta4 = pi - acos((l1 ** 2 + l2 ** 2 - distance) / (2 * l1 * l2))
    #
    #         try:
    #             theta6 = atan((T_i[1][3]) / (T_i[2][3]))
    #         except:
    #             theta6 = 0
    #
    #         T_5_6 = [[cos(theta6), -sin(theta6), 0, 0],
    #                  [sin(theta6) * cos(-pi / 2), cos(theta6) * cos(-pi / 2), -sin(-pi / 2), 0],
    #                  [sin(theta6) * sin(-pi / 2), cos(theta6) * sin(-pi / 2), cos(-pi / 2), 0],
    #                  [0, 0, 0, 1]]
    #
    #         T_ii = np.linalg.pinv(np.linalg.pinv(T_i) * np.linalg.pinv(T_5_6 * R_z * R_y))
    #
    #         # T_i_inv = np.dot(R_x, T_i)
    #         # T_i = np.linalg.pinv(T_i)
    #
    #         # A_5_6 = self.transforms[]
    #         # T_5_6 =
    #         # T_ii = np.linalg.pinv( np.linalg.pinv(T_i) * np.linalg.pinv( T_5_6 * R_z * R_y ) )

    # Solve inverse kinematics by using analytical or numerical method

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''

        def set_names(chain):
            return self.chains[chain]

        def set_times(names):
            # For three different joints, times should be: [[times for joint 1], [times for joint 2], [times for joint 3]]
            # The times of each joint should be [0, 3]. Assuming the movement should start immediately and take 3 seconds to complete.
            # So times should be: [[0, 3], [0, 3], [0, 3]]
            return [[0, 3]] *  len(names)

        def set_keys(names, times, angles):
            # For three different joints, keys should be: [[keys for joint 1], [keys for joint 2], [keys for joint 3]]
            # The keys of each joint should be [key1, key2]. Assuming the movement should only be from A to B and no keys in between.
            # So keys should be: [[key1, key2], [key1, key2], [key1, key2]]
            # Each key is [float, [int, float, float], [int, float, float]
            # So keys at the end should be: [[[angle1, handle11, handle12], [angle2, handle21, handle22]] * 3]
            temp_keys = []
            temp_key = []
            for i, joint in enumerate(names):
                for time in times[i]:
                    if time == 0:
                        angle = self.perception.joint[joint]
                        handle1 = [3, 0, 0]
                        handle2 = [3, 0, 0]
                        key0 = [angle, handle1, handle2]
                        temp_key.append(key0)
                    else:
                        angle = angles[joint]
                        handle1 = [3, 0, 0]
                        handle2 = [3, 0, 0]
                        key1 = [angle, handle1, handle2]
                        temp_key.append(key1)
                temp_keys.append(temp_key)
                temp_key = []
            return temp_keys

        # Calculate joint angles using inverse kinematic: transform -> joint_angles
        joint_angles = self.inverse_kinematics(effector_name, transform)

        # Build the keyframes that the robot is going to have to follow
        names = set_names(effector_name)
        times = set_times(names)
        keys = set_keys(names, times, joint_angles)

        # Store the keyframes in the agent
        self.keyframes = (names, times, keys)
        # maybe keyframes should be a list not a tuple but in the vorgabe in was a tuple


if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    # T[-1, 0] = -0.1
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26
    T = T.T
    agent.set_transforms('LLeg', T)
    agent.run()
