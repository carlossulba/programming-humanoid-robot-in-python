'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    1. the local_trans has to consider different joint axes and link parameters for different joints
    2. Please use radians and meters as unit.
'''

# add PYTHONPATH
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))

from numpy.matlib import matrix, identity

from recognize_posture import PostureRecognitionAgent


class ForwardKinematicsAgent(PostureRecognitionAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {'Head': ['HeadYaw', 'HeadPitch'],
                       'LArm': ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll'],
                       'RArm': ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll'],
                       'LLeg': ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll'],
                       'RLeg': ['RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
                       }
        self.joint_lengths = {'HeadYaw': (0.00, 0.00, 0.1265),
                              'HeadPitch': (0.00, 0.00, 0.00),

                              'LShoulderPitch': (0.00, 0.098, 0.1),
                              'LShoulderRoll': (0.00, 0.00, 0.00),
                              'LElbowYaw': (0.105, 0.015, 0.00),
                              'LElbowRoll': (0.00, 0.00, 0.00),

                              'RShoulderPitch': (0.00, -0.098, 0.1),
                              'RShoulderRoll': (0.00, 0.00, 0.00),
                              'RElbowYaw': (0.105, -0.015, 0.00),
                              'RElbowRoll': (0.00, 0.00, 0.00),

                              'LHipYawPitch': (0.00, 0.05, -0.085),
                              'LHipRoll': (0.00, 0.00, 0.00),
                              'LHipPitch': (0.00, 0.00, 0.00),
                              'LKneePitch': (0.00, 0.00, -0.1),
                              'LAnklePitch': (0.00, 0.00, -0.1029),
                              'LAnkleRoll': (0.00, 0.00, 0.00),

                              'RHipYawPitch': (0.00, -0.05, -0.085),
                              'RHipRoll': (0.00, 0.00, 0.00),
                              'RHipPitch': (0.00, 0.00, 0.00),
                              'RKneePitch': (0.00, 0.00, -0.1),
                              'RAnklePitch': (0.00, 0.00, -0.1029),
                              'RAnkleRoll': (0.00, 0.00, 0.00),
                              }

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, joint_angle):
        '''calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        '''

        def rotation_x(angle):
            """Create a rotation matrix for a rotation around the x-axis."""
            cos, sin = np.cos(angle), np.sin(angle)
            return np.array([[1, 0, 0, 0],
                             [0, cos, -sin, 0],
                             [0, sin, cos, 0],
                             [0, 0, 0, 1]])

        def rotation_y(angle):
            """Create a rotation matrix for a rotation around the y-axis."""
            cos, sin = np.cos(angle), np.sin(angle)
            return np.array([[cos, 0, sin, 0],
                             [0, 1, 0, 0],
                             [-sin, 0, cos, 0],
                             [0, 0, 0, 1]])

        def rotation_z(angle):
            """Create a rotation matrix for a rotation around the z-axis."""
            cos, sin = np.cos(angle), np.sin(angle)
            return np.array([[cos, sin, 0, 0],
                             [-sin, cos, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

        T = identity(4)
        # YOUR CODE HERE
        if 'Roll' in joint_name:
            T = rotation_x(joint_angle)
        elif 'Pitch' in joint_name:
            T = rotation_y(joint_angle)
        elif 'Yaw' in joint_name:
            T = rotation_z(joint_angle)

        # Store the offsets in the last column.
        T[0:3, -1] = self.joint_lengths[joint_name]
        # Where are the offsets of each joint? Last row or last column? I think column
        return T

    def forward_kinematics(self, joints):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        for chain_joints in self.chains.values():
            T = identity(4)
            for joint_name in chain_joints:
                if joint_name in joints.keys():
                    joint_angle = joints[joint_name]
                    Tl = self.local_trans(joint_name, joint_angle)
                    T = np.dot(T, Tl)
                    self.transforms[joint_name] = T


if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    agent.run()
