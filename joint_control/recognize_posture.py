'''In this exercise you need to use the learned classifier to recognize current posture of robot

* Tasks:
    1. load learned classifier in `PostureRecognitionAgent.__init__`
    2. recognize current posture in `PostureRecognitionAgent.recognize_posture`

* Hints:
    Let the robot execute different keyframes, and recognize these postures.

'''

from angle_interpolation import AngleInterpolationAgent
from keyframes import hello, leftBackToStand, leftBellyToStand, rightBackToStand, rightBellyToStand, wipe_forehead
from joblib import load


class PostureRecognitionAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PostureRecognitionAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.posture = 'unknown'
        self.posture_classifier = load('robot_pose.joblib')  # LOAD YOUR CLASSIFIER

    def think(self, perception):
        self.posture = self.recognize_posture(perception)
        return super(PostureRecognitionAgent, self).think(perception)

    def recognize_posture(self, perception):
        posture = 'unknown'
        # YOUR CODE HERE the features are['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'RHipYawPitch',
        # 'RHipRoll', 'RHipPitch', 'RKneePitch', 'AngleX', 'AngleY']
        current_state = [
            [perception.joint.get('LHipYawPitch'), perception.joint.get('LHipRoll'), perception.joint.get('LHipPitch'),
             perception.joint.get('LKneePitch'), perception.joint.get('RHipYawPitch'), perception.joint.get('RHipRoll'),
             perception.joint.get('RHipPitch'), perception.joint.get('RKneePitch'), perception.imu[0],
             perception.imu[1]]]
        posture = self.posture_classifier.predict(current_state)

        postures = {
            0: 'Back',
            1: 'Belly',
            2: 'Crouch',
            3: 'Frog',
            4: 'HeadBack',
            5: 'Knee',
            6: 'Left',
            7: 'Right',
            8: 'Sit',
            9: 'Stand',
            10: 'StandInit'
        }
        # print(postures[posture[0]])
        return postures[posture[0]]


if __name__ == '__main__':
    agent = PostureRecognitionAgent()
    agent.keyframes = leftBackToStand()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
