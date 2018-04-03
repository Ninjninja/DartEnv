import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from gym.envs.dart.push_window import *
from pydart2.gui.trackball import Trackball
class DartBlockPushEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.track_skeleton_id = -1
        control_bounds = np.array([[1.0,1.0,0,0],[-1.0,-1.0,1,3]])
        self.action_scale = 10
        dart_env.DartEnv.__init__(self, 'arti_data.skel', frame_skip = 3, observation_size=4, action_bounds=control_bounds)
        utils.EzPickle.__init__(self)

    # def do_simulation(self, tau, n_frames):
    #     for _ in range(n_frames):
    #         skel = self.robot_skeleton[-1]
    #         bod = skel.root_bodynode()
    #         bod.add_ext_force(np.array([0, 0, tau[0]]), np.array([0, 0, 0]))
    def do_simulation(self, tau, n_frames):
        # self.robot_skeleton.set_forces(tau)
        skel = self.robot_skeleton
        bod = skel.bodynodes[0]
        bod.add_ext_force(np.array([0, 0, 0.5]), np.array([tau[0], 0, 0]))
        bod = skel.bodynodes[1]
        bod.add_ext_force(np.array([0, 0, 0.5]), np.array([tau[1], 0, 0]))
        for _ in range(n_frames):
            self.dart_world.step()


    def _step(self, a):
        # reward = 1.0
        # print(a)
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0] = a[0] * self.action_scale
        tau[1] = a[1] * self.action_scale
        mass_1 = a[2]
        is_predicting = a[3]
        body_mass = self.robot_skeleton.bodynodes[0].m+ self.robot_skeleton.bodynodes[1].m
        if not is_predicting:
            self.do_simulation(tau, self.frame_skip)
            ob = self._get_obs()
            reward = -1
        else:
            ob = self._get_obs()
            error = (mass_1 - body_mass)/body_mass
            if error<0.1:
                reward = 10
            else:
                reward = -10
            # notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
            # done = not notdone
            # print(' '+str(self.dart_world.t))
        if self.dt >0.5:
            done = 1
        else:
            done = 0
        return ob, reward, done, {}


    def _get_obs(self):
        return self._get_viewer().getFrame()
        #return np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()

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
    def getViewer(self, sim, title=None):
        # glutInit(sys.argv)
        win = PushGLUTWindow(sim,title)
        win.scene.add_camera(Trackball(theta=-45.0, phi = 0.0, zoom=0.1), 'gym_camera')
        win.scene.set_camera(win.scene.num_cameras()-1)
        win.run()
        return win

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = self.getViewer(self.dart_world)
            self.viewer_setup()
        return self.viewer