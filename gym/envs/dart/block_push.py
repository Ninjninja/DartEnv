import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from gym.envs.dart.static_window import *

class DartBlockPushEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.track_skeleton_id = -1
        control_bounds = np.array([[1.0,1.0],[-1.0,-1.0]])
        self.action_scale = 1
        dart_env.DartEnv.__init__(self, 'arti_data.skel', frame_skip = 2, observation_size=4, action_bounds=control_bounds)
        utils.EzPickle.__init__(self)

    # def do_simulation(self, tau, n_frames):
    #     for _ in range(n_frames):
    #         skel = self.robot_skeleton[-1]
    #         bod = skel.root_bodynode()
    #         bod.add_ext_force(np.array([0, 0, tau[0]]), np.array([0, 0, 0]))
    def do_simulation(self, tau, n_frames):
        for _ in range(n_frames):
            # self.robot_skeleton.set_forces(tau)
            skel = self.robot_skeleton
            bod = skel.bodynodes[0]
            bod.add_ext_force(np.array([0, 0, 0.5]), np.array([tau[0], 0, 0]))
            bod = skel.bodynodes[1]
            bod.add_ext_force(np.array([0, 0, 0.5]), np.array([tau[1], 0, 0]))
            self.dart_world.step()


    def _step(self, a):
        reward = 1.0
        print(a)
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0] = a[0] * self.action_scale
        tau[1] = a[1] * self.action_scale
        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()

        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}


    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -0.5
        self._get_viewer().scene.tb.trans[1] = 0.2
        self._get_viewer().scene.tb._set_theta(-45)
        self.track_skeleton_id = 0
