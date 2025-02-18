'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''
import sys

from pid import PIDAgent
from keyframes import hello, leftBackToStand, leftBellyToStand, rightBackToStand, rightBellyToStand, wipe_forehead


def calc_bezier(t, p0, p1, p2, p3):
    return (
            ((1 - t) ** 3 * p0)
            + (3 * (1 - t) ** 2 * t * p1)
            + (3 * (1 - t) * (t ** 2) * p2)
            + ((t ** 3) * p3)
    )


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])
        self.born_time = self.perception.time
        self.control = -1
        self.max_time = -1

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}

        if self.keyframes == ([], [], []):
            return target_joints

        if self.control == -1:
            self.control = 1
            self.born_time = self.perception.time
            self.max_time = -1

        current_time = perception.time - self.born_time

        names, times, keys = keyframes

        for i, name in enumerate(names):
            # Find the correct time interval
            for j in range(len(times[i]) - 1):
                if times[i][-1] > self.max_time:
                    self.max_time = times[i][-1]

                if times[i][j] <= current_time <= times[i][j + 1]:
                    # Calculate Bezier control points
                    p0 = keys[i][j][0]
                    p1 = p0 + keys[i][j][2][2]
                    p3 = keys[i][j + 1][0]
                    p2 = p3 + keys[i][j + 1][1][2]
                    t = (current_time - times[i][j]) / (times[i][j + 1] - times[i][j])
                    target_joints[name] = calc_bezier(t, p0, p1, p2, p3)
                    break
                elif current_time < times[i][0]:
                    # In case we are before the first keyframe
                    if name in perception.joint:
                        p0 = perception.joint[name]
                    else:
                        p0 = keys[i][0][0]
                    p3 = keys[i][0][0]

                    p1 = p0 + 0
                    p2 = p3 + 0
                    t = current_time / times[i][0]
                    target_joints[name] = calc_bezier(t, p0, p1, p2, p3)
                    break

        # Copy joint angle from one side to the other if necessary
        if 'LHipYawPitch' in target_joints:
            target_joints['RHipYawPitch'] = target_joints['LHipYawPitch']

        if current_time > self.max_time:
            self.keyframes = ([], [], [])
            self.control = -1

        return target_joints


if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
