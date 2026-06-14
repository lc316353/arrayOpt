# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:53:42 2024

@author: Patrick Schillings
based on code by Francesca Badaracco

page references refer to the PhD thesis of Francesca Badaracco (2021):
Newtonian Noise Studies in 2nd and 3rd Generation Gravitational-Wave Interferometric Detectors

"""

from pyswarms.single.general_optimizer import GeneralOptimizerPSO
from pyswarms.backend.topology import Ring

from scipy.optimize import differential_evolution

import json
import ast

import numpy as np
from scipy import special as sp
import time
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)

import jax_bessel
from jax.scipy import linalg

import jax.numpy as jnp
import optax

from functools import partial


energyline = 16
stateline = 21
timeline = 35
blocksize = 35

pline = 10
SNRline = 11
fline = 13
boundline = 14
titleline = 3

pi = np.pi

e1_Tri = np.array([1, 0, 0])
e2_Tri = np.array([0.5, np.sqrt(3) / 2, 0])
e1_sym = np.array([np.sqrt(3) / 2, -0.5, 0])
e2_sym = np.array([np.sqrt(3) / 2, 0.5, 0])
e1_L = np.array([1, 0, 0])
e2_L = np.array([0, 1, 0])


def PSO_wrapper(PSO_state, func, args):
    swarmSize = PSO_state.shape[0]
    Res_Vec = np.zeros(swarmSize)
    for tt in range(swarmSize):
        Res_Vec[tt] = func(PSO_state[tt], *args)
    return Res_Vec


class AnalyticResidual:
    """
    A class that defines the Newtonian noise residual for a seismometer array and several parameters.
    It can be used to optimize the seismometer positions via the optimization algorithm described in
    https://iopscience.iop.org/article/10.1088/1361-6382/adb898

    All spectral density functions and the residual are implemented with JAX so that they are
    differentiable and JIT-compilable. This enables gradient-based optimization (Adam) in addition
    to the population-based methods (PSO, DE). Optimizers can be chained via optimize_chain().

    Parameters
    ----------
    default_mode : string, optional
        The coordinate system the state should be transferred into. The default is "volume".
        Possible modes:
            "volume"                  cartesian 3D
            "sphere"                  spherical 2D at radius tunnel_radius
            "cylinder"                cylindrical 2D along x-axis at radius tunnel_radius
            "2cylinder"               cylindrical 2D along both arms (phi > 0 → arm 1, phi < 0 → arm 2)
            "2cylindervolume"         cylindrical 3D along both arms
            "volume forcesym"         extra seismometer mirrored at bisecting plane
            "volume forcesym mirror"  extra seismometer mirrored at bisecting line
            "volume multipleX"        X seismometers per borehole
    default_loss : string, optional
        Output of the residual function. The default is "mean".
        Options: "in", "in1", "in2", "end1", "end2", "all"/"max", "mean"
    e1, e2 : array of length 3, optional
        Unit vectors along the two arms. The default is e1_Tri, e2_Tri.
    d_in1, d_in2 : float, optional
        Distance from corner to in mirrors [m]. The default is 64.12 m.
    d_end1, d_end2 : float, optional
        Distance from corner to end mirrors [m]. The default is 536.35 m.
    c_p : float, optional
        P-wave speed [m/s]. The default is 6000 m/s.
    c_s : float, optional
        S-wave speed [m/s]. The default is 4000 m/s.
    tunnel_length : float, optional
        Boundary tunnel length for optimization [m]. The default is 5000 m.
    reverse_tunnel_length : float, optional
        Tunnel length on the opposite side of the corner [m]. The default is 500 m.
    tunnel_radius : float, optional
        Surface radius for 2D modes [m]. The default is 5 m.
    tunnel_radius_min, tunnel_radius_max : float, optional
        Radius bounds for "2cylindervolume" mode [m]. The defaults are 5 m and 20 m.
    polar_angle_max : float, optional
        Polar angle boundary [°]. The default is 380°.
    azimuthal_angle_max : float, optional
        Azimuthal angle boundary [°] for "sphere" mode. The default is 200°.
    volume_extent_xy : list [min, max], optional
        x/y coordinate bounds for "volume" modes [m]. The default is [-1000, 1000].
    volume_extent_z : list [min, max], optional
        z coordinate bounds for "volume" modes [m]. The default is [-300, 300].
    """

    e1 = np.array([1, 0, 0])
    e2 = np.array([0.5, np.sqrt(3) / 2, 0])

    d_in1 = 64.12
    d_in2 = 64.12
    d_end1 = 536.35
    d_end2 = 536.35

    c_p = 6000
    c_s = 4000

    tunnel_length = 5000
    reverse_tunnel_length = 500
    tunnel_radius = 5
    tunnel_radius_min = 5
    tunnel_radius_max = 20
    polar_angle_max = 380
    azimuthal_angle_max = 200
    volume_extent_xy = [-1000, 1000]
    volume_extent_z = [-300, 300]

    default_mode = "volume"
    default_loss = "mean"
    mode = "volume"
    loss = "mean"
    optimization_method = "particleSwarm"

    Nmult = 1
    dim = 3
    lower_bound = [volume_extent_xy[0], volume_extent_xy[0], volume_extent_z[0]]
    upper_bound = [volume_extent_xy[1], volume_extent_xy[1], volume_extent_z[1]]

    def __init__(self, default_mode="volume", default_loss="mean", e1=e1_Tri, e2=e2_Tri,
                 d_in1=64.12, d_in2=64.12, d_end1=536.35, d_end2=536.35,
                 c_p=6000, c_s=4000, tunnel_length=5000, reverse_tunnel_length=500,
                 tunnel_radius=5, tunnel_radius_min=5, tunnel_radius_max=20,
                 polar_angle_max=380, azimuthal_angle_max=200,
                 volume_extent_xy=None, volume_extent_z=None):

        if volume_extent_xy is None:
            volume_extent_xy = [-1000, 1000]
        if volume_extent_z is None:
            volume_extent_z = [-300, 300]

        self.e1 = e1
        self.e2 = e2
        self.d_in1 = d_in1
        self.d_in2 = d_in2
        self.d_end1 = d_end1
        self.d_end2 = d_end2

        self.c_p = c_p
        self.c_s = c_s

        self.tunnel_length = tunnel_length
        self.reverse_tunnel_length = reverse_tunnel_length
        self.tunnel_radius = tunnel_radius
        self.tunnel_radius_min = tunnel_radius_min
        self.tunnel_radius_max = tunnel_radius_max
        self.polar_angle_max = polar_angle_max
        self.azimuthal_angle_max = azimuthal_angle_max
        self.volume_extent_xy = volume_extent_xy
        self.volume_extent_z = volume_extent_z

        self.default_mode = default_mode
        self.default_loss = default_loss
        self.mode = default_mode
        self.loss = default_loss

        self.Nmult = self.get_Nmult()
        self.dim = self.get_dim()
        self.lower_bound, self.upper_bound = self.get_bounds()



    def defineGeometry(self, e1=None, e2=None, d_in1=None, d_in2=None, d_end1=None, d_end2=None,
                       c_p=None, c_s=None, tunnel_length=None, reverse_tunnel_length=None,
                       tunnel_radius=None, tunnel_radius_min=None, tunnel_radius_max=None,
                       polar_angle_max=None, azimuthal_angle_max=None,
                       volume_extent_xy=None, volume_extent_z=None):
        """Redefine geometry parameters. Any parameter left as None keeps its current value."""

        self.e1 = self.e1 if e1 is None else e1
        self.e2 = self.e2 if e2 is None else e2
        self.d_in1 = self.d_in1 if d_in1 is None else d_in1
        self.d_in2 = self.d_in2 if d_in2 is None else d_in2
        self.d_end1 = self.d_end1 if d_end1 is None else d_end1
        self.d_end2 = self.d_end2 if d_end2 is None else d_end2

        self.c_p = self.c_p if c_p is None else c_p
        self.c_s = self.c_s if c_s is None else c_s

        self.tunnel_length = self.tunnel_length if tunnel_length is None else tunnel_length
        self.reverse_tunnel_length = self.reverse_tunnel_length if reverse_tunnel_length is None else reverse_tunnel_length
        self.tunnel_radius = self.tunnel_radius if tunnel_radius is None else tunnel_radius
        self.tunnel_radius_min = self.tunnel_radius_min if tunnel_radius_min is None else tunnel_radius_min
        self.tunnel_radius_max = self.tunnel_radius_max if tunnel_radius_max is None else tunnel_radius_max
        self.polar_angle_max = self.polar_angle_max if polar_angle_max is None else polar_angle_max
        self.azimuthal_angle_max = self.azimuthal_angle_max if azimuthal_angle_max is None else azimuthal_angle_max
        self.volume_extent_xy = self.volume_extent_xy if volume_extent_xy is None else volume_extent_xy
        self.volume_extent_z = self.volume_extent_z if volume_extent_z is None else volume_extent_z

        self.Nmult = self.get_Nmult()
        self.dim = self.get_dim()
        self.lower_bound, self.upper_bound = self.get_bounds()



    def set_default_mode(self, mode):
        """Sets the default coordinate system mode."""
        self.default_mode = mode
        self.mode = mode
        self.Nmult = self.get_Nmult()
        self.dim = self.get_dim()
        self.lower_bound, self.upper_bound = self.get_bounds()



    def set_default_loss(self, loss):
        """Sets the default optimization loss."""
        self.default_loss = loss
        self.loss = loss



    def get_Nmult(self, mode=None):
        """Returns the number of seismometers per borehole X in mode "volume multipleX"."""
        if mode is None:
            mode = self.mode
        if "multiple" in mode.lower():
            try:
                return int(mode.split("multiple")[-1])
            except Exception:
                print("Could not set Nmult!")
                return 1
        return 1



    def get_dim(self, mode=None):
        """Returns the number of parameters describing one seismometer/borehole in the given mode."""
        if mode is None:
            mode = self.mode
            Nmult = self.Nmult
        else:
            Nmult = self.get_Nmult(mode)

        if "volume" in mode.lower():
            return 3 + Nmult - 1
        else:
            return 2



    def get_bounds(self, mode=None, tunnel_length=None, reverse_tunnel_length=None,
                   tunnel_radius_min=None, tunnel_radius_max=None,
                   polar_angle_max=None, azimuthal_angle_max=None,
                   volume_extent_xy=None, volume_extent_z=None):
        """Returns (lower_bound, upper_bound) lists for the given mode."""
        if mode is None:
            mode = self.mode
            Nmult = self.Nmult
            dim = self.dim
        else:
            Nmult = self.get_Nmult(mode)
            dim = self.get_dim(mode)

        if tunnel_length is None:
            tunnel_length = self.tunnel_length
        if reverse_tunnel_length is None:
            reverse_tunnel_length = self.reverse_tunnel_length
        if tunnel_radius_min is None:
            tunnel_radius_min = self.tunnel_radius_min
        if tunnel_radius_max is None:
            tunnel_radius_max = self.tunnel_radius_max
        if polar_angle_max is None:
            polar_angle_max = self.polar_angle_max
        if azimuthal_angle_max is None:
            azimuthal_angle_max = self.azimuthal_angle_max
        if volume_extent_xy is None:
            volume_extent_xy = self.volume_extent_xy
        if volume_extent_z is None:
            volume_extent_z = self.volume_extent_z

        if dim == 2 and "sphere" in mode.lower():
            lower_bound = [0, 0]
            upper_bound = [polar_angle_max, azimuthal_angle_max]
        elif dim == 2 and "cylinder" in mode.lower() and "2" not in mode:
            lower_bound = [0, -reverse_tunnel_length]
            upper_bound = [polar_angle_max, tunnel_length]
        elif dim == 2 and "cylinder" in mode.lower() and "2" in mode:
            lower_bound = [-polar_angle_max, -reverse_tunnel_length]
            upper_bound = [polar_angle_max, tunnel_length]
        elif dim == 3 and "cylinder" in mode.lower() and "2" in mode:
            lower_bound = [-polar_angle_max, -reverse_tunnel_length, tunnel_radius_min]
            upper_bound = [polar_angle_max, tunnel_length, tunnel_radius_max]
        elif dim == 3:
            lower_bound = [volume_extent_xy[0], volume_extent_xy[0], volume_extent_z[0]]
            upper_bound = [volume_extent_xy[1], volume_extent_xy[1], volume_extent_z[1]]
        elif dim > 3:
            lower_bound = [volume_extent_xy[0], volume_extent_xy[0]] + [volume_extent_z[0]] * Nmult
            upper_bound = [volume_extent_xy[1], volume_extent_xy[1]] + [volume_extent_z[1]] * Nmult

        return lower_bound, upper_bound



    def state_to_coordinates(self, state, N, mode=None):
        """
        Returns Cartesian coordinates of a state given the mode.

        Parameters
        ----------
        state : array of size N*dim
            Seismometer coordinates in the current mode's parameterization.
        N : int
            Number of optimizable coordinate sets.
        mode : string, optional
            Coordinate mode. Defaults to self.mode.

        Returns
        -------
        x, y, z : jax arrays of length Nloc
        Nloc : int
            Total number of seismometers (may exceed N for forcesym/multipleX modes).
        """
        if mode is None:
            mode = self.mode
            dim = self.dim
            Nmult = self.Nmult
        else:
            Nmult = self.get_Nmult(mode)
            dim = self.get_dim(mode)

        state = jnp.array(state, dtype=jnp.float64)
        s = state.reshape(N, dim)
        Nloc = N

        if dim == 2 and "sphere" in mode.lower():
            phi = s[:, 0] * pi / 180
            theta = s[:, 1] * pi / 180
            x = self.tunnel_radius * jnp.sin(theta) * jnp.cos(phi)
            y = self.tunnel_radius * jnp.sin(theta) * jnp.sin(phi)
            z = self.tunnel_radius * jnp.cos(theta)

        elif dim == 2 and "cylinder" in mode.lower() and "2" not in mode:
            phi = s[:, 0] * pi / 180
            x = s[:, 1]
            y = self.tunnel_radius * jnp.cos(phi)
            z = self.tunnel_radius * jnp.sin(phi)

        elif dim == 2 and "cylinder" in mode.lower() and "2" in mode:
            # phi > 0 → arm 1 (e1), phi <= 0 → arm 2 (e2)
            e_perp1_x = np.cross(self.e1, np.array([0, 0, 1]))[0]
            e_perp1_y = np.cross(self.e1, np.array([0, 0, 1]))[1]
            e_perp2_x = np.cross(self.e2, np.array([0, 0, 1]))[0]
            e_perp2_y = np.cross(self.e2, np.array([0, 0, 1]))[1]
            pos = s[:, 0] > 0
            arm_x = self.e1[0] * pos + self.e2[0] * (~pos)
            arm_y = self.e1[1] * pos + self.e2[1] * (~pos)
            e_px = e_perp1_x * pos - e_perp2_x * (~pos)
            e_py = e_perp1_y * pos - e_perp2_y * (~pos)
            phi = jnp.abs(s[:, 0]) * pi / 180
            x = s[:, 1] * arm_x + self.tunnel_radius * jnp.cos(phi) * e_px
            y = s[:, 1] * arm_y + self.tunnel_radius * jnp.cos(phi) * e_py
            z = self.tunnel_radius * jnp.sin(phi)

        elif dim == 3 and "cylinder" in mode.lower() and "2" in mode:
            e_perp1_x = np.cross(self.e1, np.array([0, 0, 1]))[0]
            e_perp1_y = np.cross(self.e1, np.array([0, 0, 1]))[1]
            e_perp2_x = np.cross(self.e2, np.array([0, 0, 1]))[0]
            e_perp2_y = np.cross(self.e2, np.array([0, 0, 1]))[1]
            pos = s[:, 0] > 0
            arm_x = self.e1[0] * pos + self.e2[0] * (~pos)
            arm_y = self.e1[1] * pos + self.e2[1] * (~pos)
            e_px = e_perp1_x * pos - e_perp2_x * (~pos)
            e_py = e_perp1_y * pos - e_perp2_y * (~pos)
            phi = jnp.abs(s[:, 0]) * pi / 180
            x = s[:, 1] * arm_x + s[:, 2] * jnp.cos(phi) * e_px
            y = s[:, 1] * arm_y + s[:, 2] * jnp.cos(phi) * e_py
            z = s[:, 2] * jnp.sin(phi)

        elif dim == 3:
            x = s[:, 0]
            y = s[:, 1]
            z = s[:, 2]

        else:  # dim > 3: "volume multipleX"
            x = jnp.tile(s[:, 0], Nmult)
            y = jnp.tile(s[:, 1], Nmult)
            z = jnp.ravel(s[:, 2:].T)
            Nloc = N * Nmult

        if "forcesym" in mode.lower():
            Nloc = 2 * N
            x = jnp.concatenate((x, x))
            y = jnp.concatenate((y, -y))
            if "mirror" in mode.lower():
                z = jnp.concatenate((z, -z))
            else:
                z = jnp.concatenate((z, z))

        return x, y, z, Nloc



    def multiple_to_volume_state(self, state, multiple):
        """Returns the equivalent "volume" state for a "volume multipleX" state."""
        state = np.array(state)
        N = np.prod(state.shape) // (2 + multiple)
        state = state.reshape((N, 2 + multiple))
        newstate = np.zeros((N * multiple, 3))
        newstate[:, :2] = np.repeat(state[:, :2], multiple, axis=0)
        newstate[:, 2] = state[:, 2:].flatten()
        return newstate



    # ************************** SPECTRAL DENSITIES ************************** #

    @partial(jax.jit, static_argnums=(0, 4,))
    def css(self, x, y, z, N, freq, SNR, p):
        """
        Cross power spectral density matrix Css of all seismometer channels (3N x 3N).

        Parameters
        ----------
        x, y, z : arrays of length N
            Seismometer Cartesian coordinates.
        N : int (static)
            Number of seismometers.
        freq : float
            Wiener filter frequency.
        SNR : float
            Signal-to-noise ratio (sets diagonal regularisation).
        p : float
            P-wave fraction in [0, 1].

        Returns
        -------
        Css : (3N, 3N) array
        """
        kp = 2 * pi * freq / self.c_p
        ks = 2 * pi * freq / self.c_s

        meshgridx = jnp.meshgrid(x, x)
        meshgridy = jnp.meshgrid(y, y)
        meshgridz = jnp.meshgrid(z, z)

        mx = meshgridx[1] - meshgridx[0]
        my = meshgridy[1] - meshgridy[0]
        mz = meshgridz[1] - meshgridz[0]

        # 1e-30 avoids sqrt(0) in backward pass (diagonal is always dist=0).
        dist = jnp.sqrt(mx ** 2 + my ** 2 + mz ** 2 + 1e-30)

        mx = mx / dist
        my = my / dist
        mz = mz / dist

        zo = jnp.zeros((N, N))
        o = jnp.ones((N, N))
        e1DoTe2 = jnp.concatenate((jnp.concatenate((o, zo, zo), 1),
                                    jnp.concatenate((zo, o, zo), 1),
                                    jnp.concatenate((zo, zo, o), 1)), 0)

        e1DoTe12 = jnp.concatenate((jnp.concatenate((mx, mx, mx), 1),
                                     jnp.concatenate((my, my, my), 1),
                                     jnp.concatenate((mz, mz, mz), 1)), 0)
        e2DoTe12 = jnp.concatenate((jnp.concatenate((mx, mx, mx), 0),
                                     jnp.concatenate((my, my, my), 0),
                                     jnp.concatenate((mz, mz, mz), 0)), 1)

        tmp = jnp.concatenate((dist, dist, dist), 1)
        dist_m = jnp.concatenate((tmp, tmp, tmp), 0)

        fp = ((jax_bessel.spherical_jv(0, dist_m * kp) + jax_bessel.spherical_jv(2, dist_m * kp)) * e1DoTe2
              - 3.0 * jax_bessel.spherical_jv(2, dist_m * kp) * e1DoTe12 * e2DoTe12)
        fs = ((jax_bessel.spherical_jv(0, dist_m * ks) - 0.5 * jax_bessel.spherical_jv(2, dist_m * ks)) * e1DoTe2
              + 1.5 * jax_bessel.spherical_jv(2, dist_m * ks) * e1DoTe12 * e2DoTe12)

        Css = p * fp + (1 - p) * fs
        dd = jnp.diag(jnp.ones(Css.shape[1], dtype=bool))
        Css = jnp.where(dd, 1 + 1 / SNR ** 2, Css)

        return Css



    @partial(jax.jit, static_argnums=(0, 4,))
    def csn_corr_funct(self, x, y, z, N, freq, e_TM, d_TM):
        """
        Helper: per-wave-type seismometer–mirror cross-PSD for a single mirror.

        Returns
        -------
        fp_s1, fs_s1 : arrays of length 3N
            P-wave and S-wave contributions to Csn.
        """
        kp = 2 * pi * freq / self.c_p
        ks = 2 * pi * freq / self.c_s

        xx = jnp.concatenate((x, x, x), axis=0)
        yy = jnp.concatenate((y, y, y), axis=0)
        zz = jnp.concatenate((z, z, z), axis=0)

        mes = jnp.zeros((3, 3 * N))
        mes = mes.at[0, 0:N].set(jnp.ones(N))
        mes = mes.at[1, N:2 * N].set(jnp.ones(N))
        mes = mes.at[2, 2 * N:3 * N].set(jnp.ones(N))

        e1 = jnp.array(e_TM, dtype=jnp.float64)

        es1 = jnp.stack([xx - d_TM * e1[0],
                          yy - d_TM * e1[1],
                          zz - d_TM * e1[2]])

        dist1 = jnp.sqrt(jnp.sum(es1 ** 2, 0) + 1e-30)
        es1 = es1 / dist1

        me1 = jnp.ones((3, 3 * N))
        me1 = me1.at[0, :].set(me1[0, :] * e1[0])
        me1 = me1.at[1, :].set(me1[1, :] * e1[1])
        me1 = me1.at[2, :].set(me1[2, :] * e1[2])

        esDoTes1 = jnp.sum(mes * es1, 0)
        e1DoTes1 = jnp.sum(me1 * es1, 0)

        esDoTe1 = e1  # scalar products with sensor axes

        def block(idx, e_comp):
            sl = slice(idx * N, (idx + 1) * N)
            fp = ((jax_bessel.spherical_jv(0, dist1[sl] * kp) + jax_bessel.spherical_jv(2, dist1[sl] * kp)) * e_comp
                  - 3.0 * jax_bessel.spherical_jv(2, dist1[sl] * kp) * esDoTes1[sl] * e1DoTes1[sl])
            fs = ((jax_bessel.spherical_jv(0, dist1[sl] * ks) - 0.5 * jax_bessel.spherical_jv(2, dist1[sl] * ks)) * e_comp
                  + 1.5 * jax_bessel.spherical_jv(2, dist1[sl] * ks) * esDoTes1[sl] * e1DoTes1[sl])
            return fp, fs

        fp0, fs0 = block(0, esDoTe1[0])
        fp1, fs1 = block(1, esDoTe1[1])
        fp2, fs2 = block(2, esDoTe1[2])

        fp_s1 = jnp.concatenate([fp0, fp1, fp2])
        fs_s1 = jnp.concatenate([fs0, fs1, fs2])

        return fp_s1, fs_s1



    @partial(jax.jit, static_argnums=(0, 4,))
    def csn_end(self, x, y, z, N, freq, SNR, p, e_TM, d_TM):
        """Csn for a single end mirror."""
        fp_s1, fs_s1 = self.csn_corr_funct(x, y, z, N, freq, e_TM, d_TM)
        Csn = 1 / 3 * (2 * p * fp_s1 - (1 - p) * fs_s1)
        return Csn



    @partial(jax.jit, static_argnums=(0, 4,))
    def csn_in(self, x, y, z, N, freq, SNR, p, e_TM1, e_TM2, d_TM1, d_TM2):
        """Csn for two correlated in-mirrors (differential signal)."""
        fp_s1, fs_s1 = self.csn_corr_funct(x, y, z, N, freq, e_TM1, d_TM1)
        fp_s2, fs_s2 = self.csn_corr_funct(x, y, z, N, freq, e_TM2, d_TM2)
        Csn = 1 / 3 * (2 * p * (fp_s2 - fp_s1) - (1 - p) * (fs_s2 - fs_s1))
        return Csn



    @partial(jax.jit, static_argnums=(0,))
    def cnn_end(self, p):
        """Cnn for a single mirror."""
        return 1 / 9 * (3 * p + 1)



    @partial(jax.jit, static_argnums=(0,))
    def cnn_in(self, freq, p, e_TM1, e_TM2, d_TM1, d_TM2):
        """Cnn for two correlated in-mirrors."""
        kp = 2 * pi * freq / self.c_p
        ks = 2 * pi * freq / self.c_s

        e_TM1 = jnp.array(e_TM1, dtype=jnp.float64)
        e_TM2 = jnp.array(e_TM2, dtype=jnp.float64)

        e2DoTe1 = jnp.dot(e_TM1, e_TM2)
        e21 = d_TM1 * e_TM1 - d_TM2 * e_TM2
        dist1 = jnp.linalg.norm(e21)
        e21 = e21 / dist1
        e2DoTe21 = jnp.dot(e_TM2, e21)
        e1DoTe21 = jnp.dot(e_TM1, e21)

        fp = ((jax_bessel.spherical_jv(0, dist1 * kp) + jax_bessel.spherical_jv(2, dist1 * kp)) * e2DoTe1
              - 3.0 * jax_bessel.spherical_jv(2, dist1 * kp) * e2DoTe21 * e1DoTe21)
        fs = ((jax_bessel.spherical_jv(0, dist1 * ks) - 0.5 * jax_bessel.spherical_jv(2, dist1 * ks)) * e2DoTe1
              + 1.5 * jax_bessel.spherical_jv(2, dist1 * ks) * e2DoTe21 * e1DoTe21)

        Cnn = 1 / 9 * (2 * (3 * p + 1) - 2 * (4 * p * fp + (1 - p) * fs))
        return Cnn



    @jax.jit
    def css_svd(self, Css):
        """
        Pseudoinverse of Css via truncated SVD (threshold 0.01).

        Used as a fallback when direct solve fails. Because dynamic slicing on kVal
        is incompatible with JAX JIT, this is only intended for diagnostics outside
        the main residual computation path.
        """
        Diag = jnp.diag(Css)
        Nfact = jnp.sqrt(jnp.tensordot(Diag, Diag, axes=0))
        Css_norm = Css / Nfact

        U, diagS, V = linalg.svd(Css_norm)
        thresh = 0.01
        kVal = int(jnp.sum(diagS > thresh))

        iU = U.conjugate().T
        iV = V.conjugate().T
        Css_svd = (iV[:, :kVal]) @ jnp.diag(1 / diagS[:kVal]) @ (iU[:kVal, :])
        return Css_svd / Nfact



    @partial(jax.jit, static_argnums=(0, 4,))
    def wf(self, x, y, z, N, freq, SNR, p, e_TM=None, d_TM=None):
        """Wiener filter coefficients for a single mirror."""
        if e_TM is None:
            e_TM = self.e1
        if d_TM is None:
            d_TM = self.d_end1

        Css = self.css(x, y, z, N, freq, SNR, p)
        Csn = self.csn_end(x, y, z, N, freq, SNR, p, e_TM, d_TM)
        return jnp.dot(Csn, jnp.linalg.inv(Css))



    # ************************** RESIDUAL FUNCTION ************************** #

    @partial(jax.jit, static_argnums=(0, 2, 6, 7,))
    def residual(self, state, N, freq, SNR, p, mirror="all", combine_in=True):
        """
        Returns sqrt(residual) — the Wiener-filter noise reduction factor.

        Parameters
        ----------
        state : array of size N*dim
            Seismometer positions in the current mode's parameterization.
        N : int (static)
            Number of optimizable coordinate sets.
        freq : float
            Wiener filter frequency.
        SNR : float
            Seismometer signal-to-noise ratio.
        p : float
            P-wave fraction in [0, 1].
        mirror : string (static)
            Which mirror(s) to report: "in", "in1", "in2", "end1", "end2",
            "all"/"max" (maximum over mirrors), "mean".
        combine_in : bool (static)
            If True, treat in-mirrors as correlated (single differential signal).

        Returns
        -------
        float
            sqrt(residual), in [0, 1]; lower is better.
        """
        state = jnp.array(state, dtype=jnp.float64)
        x, y, z, Nloc = self.state_to_coordinates(state, N)

        Css = self.css(x, y, z, Nloc, freq, SNR, p)
        Csn_in_val = self.csn_in(x, y, z, Nloc, freq, SNR, p,
                                  self.e1, self.e2, self.d_in1, self.d_in2)
        Csn_in1 = self.csn_end(x, y, z, Nloc, freq, SNR, p, self.e1, self.d_in1)
        Csn_in2 = self.csn_end(x, y, z, Nloc, freq, SNR, p, self.e2, self.d_in2)
        Csn_end1 = self.csn_end(x, y, z, Nloc, freq, SNR, p, self.e1, self.d_end1)
        Csn_end2 = self.csn_end(x, y, z, Nloc, freq, SNR, p, self.e2, self.d_end2)
        Cnn_end = self.cnn_end(p)
        Cnn_in = self.cnn_in(freq, p, self.e1, self.e2, self.d_in1, self.d_in2)

        # Both branches are resolved at trace time (combine_in is static)
        if combine_in:
            Csn_list = [Csn_in_val, Csn_end1, Csn_end2]
            Cnn_list = [Cnn_in, Cnn_end, Cnn_end]
        else:
            Csn_list = [Csn_in1, Csn_end1, Csn_end2, Csn_in2]
            Cnn_list = [Cnn_end, Cnn_end, Cnn_end, Cnn_end]

        nn = len(Csn_list)
        Res_v = jnp.zeros(nn)

        for rr in range(nn):
            X = linalg.solve(Css, Csn_list[rr])
            resid = 1.0 - jnp.dot(Csn_list[rr], X) / Cnn_list[rr]
            # Clamp to zero — negative values indicate numerical near-singularity
            resid = jnp.maximum(resid, 1e-12)
            Res_v = Res_v.at[rr].set(resid)

        # Mirror selection is resolved at trace time (mirror is static)
        if "in" in mirror and combine_in:
            residual_val = Res_v[0]
        elif not combine_in and mirror == "in1":
            residual_val = Res_v[0]
        elif not combine_in and mirror == "in2":
            residual_val = Res_v[3]
        elif mirror == "end1":
            residual_val = Res_v[1]
        elif mirror == "end2":
            residual_val = Res_v[2]
        elif mirror == "all" or mirror == "max":
            residual_val = jnp.max(Res_v)
        elif mirror == "mean":
            residual_val = jnp.mean(Res_v)
        else:
            residual_val = jnp.max(Res_v)

        return jnp.sqrt(residual_val + 1e-12)



    # ************************** OPTIMIZATION ************************** #

    def optimize_PSO(self, N, freq, SNR, p, loss="all", combine_in=True,
                     optimization_options=None, worker=1, savename="",
                     step_callback=None, stop_event=None):
        """
        Optimize seismometer positions with Particle Swarm Optimization.

        Parameters
        ----------
        N : int
            Number of optimizable coordinate sets.
        freq : float
            Wiener filter frequency.
        SNR : float
            Seismometer signal-to-noise ratio.
        p : float
            P-wave fraction in [0, 1].
        loss : string, optional
            Mirror selection for the residual. The default is "all".
        combine_in : bool, optional
            Whether to treat in-mirrors as correlated. The default is True.
        optimization_options : dict, optional
            PSO hyperparameters (swarm_size, c1, c2, w, k, p, niter, ftol, ftol_iter).
        worker : int, optional
            Number of parallel workers. The default is 1.
        savename : string, optional
            If non-empty, write result to this file. The default is "".

        Returns
        -------
        (residual_value, state) : (float, array)
        """
        starttime = time.time()

        self.set_optimization_options("particleSwarm", optimization_options)
        self.loss = loss

        # Ring topology requires k < swarm_size; clamp silently.
        self.optimization_options['k'] = min(
            self.optimization_options['k'],
            self.optimization_options['swarm_size'] - 1,
        )

        x_min = np.tile(self.lower_bound, N)
        x_max = np.tile(self.upper_bound, N)
        bounds = (x_min, x_max)

        optimizer = GeneralOptimizerPSO(
            n_particles=self.optimization_options["swarm_size"],
            dimensions=self.dim * N,
            options=self.optimization_options,
            bounds=bounds,
            ftol=self.optimization_options["ftol"],
            ftol_iter=self.optimization_options["ftol_iter"],
            topology=Ring()
        )
        iteration = [0]
        _best_cost = [float("inf")]
        _best_pos  = [None]

        class _PSOStopped(Exception):
            pass

        def wrapped_pso(pso_state, func, args):
            if stop_event is not None and stop_event.is_set():
                raise _PSOStopped()
            costs = PSO_wrapper(pso_state, func, args)
            best_idx = int(np.argmin(costs))
            _best_cost[0] = float(costs[best_idx])
            _best_pos[0]  = np.array(pso_state[best_idx])
            if step_callback is not None:
                step_callback(iteration[0], _best_cost[0], _best_pos[0])
            iteration[0] += 1
            return costs

        try:
            optimizationResult = optimizer.optimize(
                wrapped_pso, self.optimization_options["niter"],
                n_processes=None, func=self.residual, args=(N, freq, SNR, p, loss, combine_in)
            )
            best_cost, best_pos = optimizationResult[0], optimizationResult[1]
        except _PSOStopped:
            print("  PSO stopped early by stop_event.")
            best_cost = _best_cost[0]
            best_pos  = _best_pos[0]
            if best_pos is None:
                best_pos = np.zeros(self.dim * N)

        if savename != "":
            self.writeFile(best_cost, best_pos, N, freq, SNR, p, savename, starttime)

        return best_cost, best_pos



    def optimize_DE(self, N, freq, SNR, p, loss="all", combine_in=True,
                    optimization_options=None, worker=1, savename="",
                    step_callback=None, stop_event=None):
        """
        Optimize seismometer positions with Differential Evolution.

        Parameters
        ----------
        N : int
            Number of optimizable coordinate sets.
        freq : float
            Wiener filter frequency.
        SNR : float
            Seismometer signal-to-noise ratio.
        p : float
            P-wave fraction in [0, 1].
        loss : string, optional
            Mirror selection for the residual. The default is "all".
        combine_in : bool, optional
            Whether to treat in-mirrors as correlated. The default is True.
        optimization_options : dict, optional
            DE hyperparameters (popsize, recombination, mutation, niter, ftol).
        worker : int, optional
            Number of parallel workers. The default is 1.
        savename : string, optional
            If non-empty, write result to this file. The default is "".

        Returns
        -------
        (residual_value, state) : (float, array)
        """
        starttime = time.time()

        self.set_optimization_options("differentialEvolution", optimization_options)
        self.loss = loss

        bound = np.array([self.lower_bound, self.upper_bound]).T
        x_bound = list(bound) * N

        residualParameter = (N, freq, SNR, p, loss, combine_in)

        iteration = [0]

        def de_cb(xk, convergence=0.0):
            if step_callback is not None:
                res = float(self.residual(
                    jnp.array(xk, dtype=jnp.float64), N, freq, SNR, p, loss, combine_in))
                step_callback(iteration[0], res, np.array(xk))
            iteration[0] += 1
            return stop_event is not None and stop_event.is_set()

        optimizationResult = differential_evolution(
            self.residual, x_bound, residualParameter,
            disp=True,
            maxiter=self.optimization_options["niter"],
            popsize=self.optimization_options["popsize"],
            init='random',
            workers=worker,
            recombination=self.optimization_options["recombination"],
            mutation=self.optimization_options["mutation"],
            strategy='best1bin',
            tol=self.optimization_options["ftol"],
            updating='deferred',
            callback=de_cb,
        )
        best_cost = float(self.residual(optimizationResult.x, *residualParameter))
        best_pos = optimizationResult.x

        if savename != "":
            self.writeFile(best_cost, best_pos, N, freq, SNR, p, savename, starttime)

        return best_cost, best_pos



    def optimize_Adam(self, N, freq, SNR, p, loss="all", combine_in=True,
                      optimization_options=None, initial_state=None, savename="",
                      step_callback=None, stop_event=None):
        """
        Optimize seismometer positions with the Adam gradient-descent algorithm.

        Uses JAX automatic differentiation through the residual function. This optimizer
        works best as a refinement step after a global search (PSO or DE), by providing
        the global result as initial_state.

        Parameters
        ----------
        N : int
            Number of optimizable coordinate sets.
        freq : float
            Wiener filter frequency.
        SNR : float
            Seismometer signal-to-noise ratio.
        p : float
            P-wave fraction in [0, 1].
        loss : string, optional
            Mirror selection for the residual. The default is "all".
        combine_in : bool, optional
            Whether to treat in-mirrors as correlated. The default is True.
        optimization_options : dict, optional
            Adam hyperparameters:
                learning_rate : step size (default 1e-5)
                max_steps     : maximum gradient steps (default 10000)
                ftol          : convergence tolerance on relative improvement (default 1e-6)
                tolerance_steps : steps over which ftol is measured (default 200)
        initial_state : array of size N*dim or None, optional
            Starting state. If None, a random state within bounds is used.
        savename : string, optional
            If non-empty, write result to this file. The default is "".
        step_callback : callable(step, residual, state) or None, optional
            Called after every gradient step with the current step index (int),
            residual (float), and state (numpy array of size N*dim). Use this to
            drive a live GUI plot. The callback runs in the optimizer thread.
        stop_event : threading.Event or None, optional
            When set, the optimization loop exits cleanly after the current step.
            Allows a GUI Stop button to interrupt a long run.

        Returns
        -------
        (residual_value, state) : (float, array of size N*dim)
        """
        starttime = time.time()

        self.set_optimization_options("adam", optimization_options)
        self.loss = loss

        opts = self.optimization_options
        learning_rate = opts["learning_rate"]
        max_steps = opts["max_steps"]
        ftol = opts["ftol"]
        tolerance_steps = opts["tolerance_steps"]

        # --- Initial state ---
        if initial_state is not None:
            state = jnp.array(initial_state, dtype=jnp.float64).ravel()
        else:
            lb = np.array(self.lower_bound)
            ub = np.array(self.upper_bound)
            state = jnp.array(
                np.random.uniform(lb, ub, size=(N, self.dim)).ravel(),
                dtype=jnp.float64
            )

        # --- Differentiable loss closure (value + gradient in one pass) ---
        def loss_fn(s):
            return self.residual(s, N, freq, SNR, p, loss, combine_in)

        vg_fn = jax.jit(jax.value_and_grad(loss_fn))

        # Warmup: trigger XLA compilation before the loop so the first step
        # is not penalised by JIT latency.
        print("  [warmup] compiling…")
        _ = vg_fn(state)
        print("  [warmup] done")

        # --- Optax Adam optimizer ---
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(state)

        init_val, _ = vg_fn(state)
        pos_hist = [state]
        res_hist = [float(init_val)]

        nsteps = 0
        prec = float("inf")

        while prec > ftol and nsteps < max_steps and not (stop_event is not None and stop_event.is_set()):
            val, grads = vg_fn(pos_hist[-1])
            res_hist.append(float(val))
            updates, opt_state = optimizer.update(grads, opt_state)
            new_pos = optax.apply_updates(pos_hist[-1], updates)
            pos_hist.append(new_pos)
            nsteps += 1

            if max_steps > 0 and nsteps % max(1, max_steps // 10) == 0:
                print(f"  Adam step {nsteps}/{max_steps}  residual={res_hist[-1]:.6f}")

            if step_callback is not None:
                step_callback(nsteps, res_hist[-1], np.array(new_pos))

            if nsteps > tolerance_steps:
                best_recent = min(res_hist[-tolerance_steps:])
                best_before = min(res_hist[:-tolerance_steps])
                if best_before > 0:
                    prec = (best_before - best_recent) / best_before
                else:
                    prec = 0.0

        best_idx = int(np.argmin(res_hist))
        best_res = res_hist[best_idx]
        best_pos = np.array(pos_hist[best_idx])

        if savename != "":
            self.writeFile(best_res, best_pos, N, freq, SNR, p, savename, starttime)

        return best_res, best_pos



    def optimize_chain(self, N, freq, SNR, p, chain, loss="all", combine_in=True, savename="",
                       step_callback=None, stop_event=None, stage_callback=None):
        """
        Run a sequence of optimizers, passing each result as the warm-start for the next.

        Population-based methods (PSO, DE) perform a global search and their result is
        passed as initial_state to any subsequent Adam step. Chaining multiple global
        methods is also supported; each runs independently and the last result is forwarded.

        Parameters
        ----------
        N : int
            Number of optimizable coordinate sets.
        freq : float
            Wiener filter frequency.
        SNR : float
            Seismometer signal-to-noise ratio.
        p : float
            P-wave fraction in [0, 1].
        chain : list of (str, dict) tuples
            Sequence of (method_name, options_dict) pairs. Supported method names:
            "PSO" / "particleSwarm", "DE" / "differentialEvolution", "Adam".
            Example:
                chain = [
                    ("DE",   {"niter": 500, "popsize": 20}),
                    ("Adam", {"max_steps": 2000, "learning_rate": 1e-4}),
                ]
        loss : string, optional
            Mirror selection for the residual. The default is "all".
        combine_in : bool, optional
            Whether to treat in-mirrors as correlated. The default is True.
        savename : string, optional
            If non-empty, write final result to this file. The default is "".
        step_callback : callable(step, residual, state) or None, optional
            Forwarded to each Adam stage. See optimize_Adam for details.
        stop_event : threading.Event or None, optional
            Forwarded to each Adam stage. When set, stops the current Adam loop.

        Returns
        -------
        (residual_value, state) : (float, array of size N*dim)
            Result of the final optimizer in the chain.
        """
        current_residual = float("inf")
        current_state = None

        for step_idx, (method, options) in enumerate(chain):
            method_lc = method.lower()
            print(f"\n--- Chain step {step_idx + 1}/{len(chain)}: {method} ---")

            if stage_callback is not None:
                stage_callback(f"Stage {step_idx + 1}/{len(chain)}: {method}")

            if stop_event is not None and stop_event.is_set():
                print("  Stop requested — aborting chain.")
                break

            if method_lc in ("pso", "particleswarm"):
                current_residual, current_state = self.optimize_PSO(
                    N, freq, SNR, p, loss=loss, combine_in=combine_in,
                    optimization_options=options,
                    step_callback=step_callback,
                    stop_event=stop_event, savename=savename
                )

            elif method_lc in ("de", "differentialevolution"):
                current_residual, current_state = self.optimize_DE(
                    N, freq, SNR, p, loss=loss, combine_in=combine_in,
                    optimization_options=options,
                    step_callback=step_callback,
                    stop_event=stop_event, savename=savename
                )

            elif method_lc == "adam":
                current_residual, current_state = self.optimize_Adam(
                    N, freq, SNR, p, loss=loss, combine_in=combine_in,
                    optimization_options=options, initial_state=current_state,
                    step_callback=step_callback, stop_event=stop_event, savename=savename
                )

            else:
                print(f"WARNING: unknown method '{method}' — skipping.")
                continue

            print(f"  -> residual after {method}: {current_residual:.6f}")

        #if savename != "" and current_state is not None:
        #    self.writeFile(current_residual, current_state, N, freq, SNR, p, savename)

        return current_residual, current_state



    def set_optimization_options(self, optimization_method=None, optimization_options=None):
        """
        Set optimizer hyperparameters.

        Parameters
        ----------
        optimization_method : string, optional
            "particleSwarm", "differentialEvolution", or "adam".
        optimization_options : dict, optional
            Overrides for the default hyperparameters of the chosen method.

            particleSwarm defaults:
                swarm_size=800, c1=1.5, c2=2, w=0.1, k=80, p=2,
                niter=1000, ftol=1e-3, ftol_iter=20
            differentialEvolution defaults:
                popsize=65, recombination=0.75, mutation=(0, 1.5),
                niter=4500, ftol=1e-3
            adam defaults:
                learning_rate=1, max_steps=10000, ftol=1e-6, tolerance_steps=200
        """
        if optimization_method is not None:
            self.optimization_method = optimization_method

        if self.optimization_method == "particleSwarm":
            self.optimization_options = {
                'swarm_size': 800, 'c1': 1.5, 'c2': 2, 'w': 0.1, 'k': 80, 'p': 2,
                'niter': 1000, 'ftol': 1e-3, 'ftol_iter': 20
            }
        elif self.optimization_method == "differentialEvolution":
            self.optimization_options = {
                'popsize': 65, 'recombination': 0.75, 'mutation': (0, 1.5),
                'niter': 4500, 'ftol': 1e-3
            }
        elif self.optimization_method == "adam":
            self.optimization_options = {
                'learning_rate': 1, 'max_steps': 10000,
                'ftol': 1e-3, 'tolerance_steps': 500
            }

        if isinstance(optimization_options, dict):
            self.optimization_options.update(optimization_options)
        elif optimization_options is not None:
            print("optimization_options must be a dict or None")



    # ************************** VISUALIZATION ************************** #

    def new_state_plot_3D(self, mirrormarker="o", mirrorcolor="r", mirrormarkersize=20, tunnelformat="--k"):
        """
        Create a new 3D figure showing mirror positions and tunnel axes.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes3D
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(self.d_in1 * self.e1[0], self.d_in1 * self.e1[1], self.d_in1 * self.e1[2],
                   c=mirrorcolor, marker=mirrormarker, s=mirrormarkersize)
        ax.scatter(self.d_end1 * self.e1[0], self.d_end1 * self.e1[1], self.d_end1 * self.e1[2],
                   c=mirrorcolor, marker=mirrormarker, s=mirrormarkersize)
        ax.scatter(self.d_in2 * self.e2[0], self.d_in2 * self.e2[1], self.d_in2 * self.e2[2],
                   c=mirrorcolor, marker=mirrormarker, s=mirrormarkersize)
        ax.scatter(self.d_end2 * self.e2[0], self.d_end2 * self.e2[1], self.d_end2 * self.e2[2],
                   c=mirrorcolor, marker=mirrormarker, s=mirrormarkersize)
        ax.plot([-self.reverse_tunnel_length * self.e1[0], self.tunnel_length * self.e1[0]],
                [-self.reverse_tunnel_length * self.e1[1], self.tunnel_length * self.e1[1]],
                tunnelformat)
        ax.plot([-self.reverse_tunnel_length * self.e2[0], self.tunnel_length * self.e2[0]],
                [-self.reverse_tunnel_length * self.e2[1], self.tunnel_length * self.e2[1]],
                tunnelformat)

        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        ax.set_zlabel(r"$z$ [m]")

        return fig, ax



    def plot_state_3D(self, ax, state, N, marker="o", color="g", markersize=10):
        """Scatter-plot a state onto an existing 3D axis."""
        x, y, z, Nloc = self.state_to_coordinates(state, N)
        ax.scatter(np.array(x), np.array(y), np.array(z), c=color, marker=marker, s=markersize)



    def writeFile(self, residual, state, N, freq, SNR, p, savename, starttime=0, directory=""):
        """Write optimization result to a text file in standard format."""
        filename = savename + '.txt'
        f = open(directory + filename, 'a+')

        f.write('\n \n \n## *************' + self.optimization_method + '-' + self.mode + '-' + self.loss
                + ': ' + str(savename) + '*************** ##\n \n \n')

        f.write('import numpy as np\n')
        f.write('import matplotlib.pyplot as plt\n')
        f.write('fig = plt.figure()\n')
        f.write("ax = fig.add_subplot(111, projection='3d')\n")

        f.write('p = '); json.dump(p, f); f.write('\n')
        f.write('SNR = '); json.dump(SNR, f); f.write('\n')
        f.write('N = '); json.dump(N, f); f.write('\n')
        f.write('f = '); json.dump(freq, f); f.write('\n')

        f.write('bounds = np.array([' + str(self.lower_bound) + ',' + str(self.upper_bound) + '])\n')
        f.write('optimization_options = ' + str(self.optimization_options) + '\n')

        f.write('Energy = ')
        json.dump(float(residual), f)

        f.write('\ne2 = '); json.dump(self.d_in2, f)
        f.write('*np.array([' + str(self.e2[0]) + ',' + str(self.e2[1]) + ',' + str(self.e2[2]) + '])')
        f.write('\ne1 = '); json.dump(self.d_in1, f)
        f.write('*np.array([' + str(self.e1[0]) + ',' + str(self.e1[1]) + ',' + str(self.e1[2]) + '])')
        f.write('\ne3 = '); json.dump(self.d_end2, f)
        f.write('*np.array([' + str(self.e2[0]) + ',' + str(self.e2[1]) + ',' + str(self.e2[2]) + '])')
        f.write('\ne4 = '); json.dump(self.d_end1, f)
        f.write('*np.array([' + str(self.e1[0]) + ',' + str(self.e1[1]) + ',' + str(self.e1[2]) + '])\n')

        state_arr = np.array(state).ravel()
        f.write('FinalState = np.array('); json.dump(state_arr.tolist(), f); f.write(')\n')
        f.write('FinalState = FinalState.reshape(N, 3)\n')
        f.write("ax.scatter(FinalState[:,0], FinalState[:,1], FinalState[:,2], c='g', marker='o')\n")
        f.write("ax.scatter(e1[0],e1[1],e1[2], c='r', marker='o')\n")
        f.write("ax.scatter(e2[0],e2[1],e2[2], c='r', marker='o')\n")
        f.write("ax.scatter(e3[0],e3[1],e3[2], c='r', marker='o')\n")
        f.write("ax.scatter(e4[0],e4[1],e4[2], c='r', marker='o')\n")
        f.write("plt.plot([0,e4[0]], [0,e4[1]], '--', c='k')\n")
        f.write("plt.plot([0,e3[0]], [0,e3[1]], '--', c='k')\n")
        f.write("ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')\n")
        f.write("ax.set_title('Energy=' + str(np.round(Energy, 4)))\n")
        f.write("plt.show()\n")
        f.write('#Finished in ' + str(np.round((time.time() - starttime) / 60, 2)) + ' minutes')

        f.close()



    def writeFile2(self, residual, state, N, freq, SNR, p, savename, starttime=0, directory=""):
        """Write optimization result in a compact key=value format."""
        with open(savename + ".txt", "a+") as f:
            f.write("mode = " + self.mode + "\n")
            f.write("N = " + str(N) + "\n")
            f.write("f = " + str(freq) + "\n")
            f.write("SNR = " + str(SNR) + "\n")
            f.write("p = " + str(p) + "\n")
            f.write("loss = " + self.loss + "\n")
            f.write("state = " + str(np.array(state).tolist()) + "\n")
            f.write("residual = " + str(float(residual)) + "\n")
            f.write("c_p = " + str(self.c_p) + "\n")
            f.write("c_s = " + str(self.c_s) + "\n")
            f.write('e_in1 = np.array([' + str(self.d_in1 * self.e1[0]) + ',' + str(self.d_in1 * self.e1[1]) + ',' + str(self.d_in1 * self.e1[2]) + '])\n')
            f.write('e_in2 = np.array([' + str(self.d_in2 * self.e2[0]) + ',' + str(self.d_in2 * self.e2[1]) + ',' + str(self.d_in2 * self.e2[2]) + '])\n')
            f.write('e_end1 = np.array([' + str(self.d_end1 * self.e1[0]) + ',' + str(self.d_end1 * self.e1[1]) + ',' + str(self.d_end1 * self.e1[2]) + '])\n')
            f.write('e_end2 = np.array([' + str(self.d_end2 * self.e2[0]) + ',' + str(self.d_end2 * self.e2[1]) + ',' + str(self.d_end2 * self.e2[2]) + '])\n')
            f.write('bounds = np.array([' + str(self.lower_bound) + ',' + str(self.upper_bound) + '])\n')
            f.write("optimization_method = " + self.optimization_method + "\n")
            f.write("//optimization_options = " + str(self.optimization_options) + "\n")
            f.write("useGPU = False\n")
            f.write("#runtime = " + str(np.round((time.time() - starttime) / 60, 2)) + " min\n")


class ReadData:

    energy = 1
    time = 0
    state = []

    N = 1
    i = 0

    f = 1
    p = 0.2
    SNR = 15
    bounds = np.array([])

    folder = "Ndependence/mean"
    tag = ""
    appendix = "Resultsall"

    optimization_method = ""
    mode = ""

    def __init__(self, tag, folder, N, i=0, appendix="Resultsall"):
        self.tag = tag
        self.N = N
        self.i = i
        self.appendix = appendix
        self.folder = folder

        try:
            energystring = str(np.loadtxt(self.folder + "/" + str(self.N) + self.tag + self.appendix + str(N) + ".txt",
                                           dtype=str, skiprows=energyline + blocksize * i, max_rows=1, delimiter="ö", comments="//"))
            self.energy = float(energystring.split("=")[-1])
            timestring = str(np.loadtxt(self.folder + "/" + str(self.N) + self.tag + self.appendix + str(N) + ".txt",
                                         dtype=str, skiprows=timeline + blocksize * i, max_rows=1, delimiter="ö", comments="//"))
            self.time = float(timestring.split(' in ')[1].split(' min')[0])
            statestring = str(np.loadtxt(self.folder + "/" + str(self.N) + self.tag + self.appendix + str(N) + ".txt",
                                          dtype=str, skiprows=stateline, max_rows=1, delimiter="ö"))
            self.state = np.array(ast.literal_eval(statestring.split("(")[1].split(")")[0]))

            pstring = str(np.loadtxt(self.folder + "/" + str(self.N) + self.tag + self.appendix + str(N) + ".txt",
                                      dtype=str, skiprows=pline + blocksize * i, max_rows=1, delimiter="ö", comments="//"))
            self.p = float(pstring.split("=")[-1])
            SNRstring = str(np.loadtxt(self.folder + "/" + str(self.N) + self.tag + self.appendix + str(N) + ".txt",
                                        dtype=str, skiprows=SNRline + blocksize * i, max_rows=1, delimiter="ö", comments="//"))
            self.SNR = float(SNRstring.split("=")[-1])
            fstring = str(np.loadtxt(self.folder + "/" + str(self.N) + self.tag + self.appendix + str(N) + ".txt",
                                      dtype=str, skiprows=fline + blocksize * i, max_rows=1, delimiter="ö", comments="//"))
            self.f = float(fstring.split("=")[-1])
            boundstring = str(np.loadtxt(self.folder + "/" + str(self.N) + self.tag + self.appendix + str(N) + ".txt",
                                          dtype=str, skiprows=boundline + blocksize * i, max_rows=1, delimiter="ö", comments="//"))
            self.bounds = np.array(ast.literal_eval(boundstring.split("(")[1].split(")")[0]))

            try:
                titlestring = str(np.loadtxt(self.folder + "/" + str(self.N) + self.tag + self.appendix + str(N) + ".txt",
                                              dtype=str, skiprows=titleline + blocksize * i, max_rows=1, delimiter="ö", comments="//"))
                self.optimization_method = titlestring.split("*************")[1].split("-")[0]
                self.mode = titlestring.split("*************")[1].split("-")[1].split(":")[0]
            except Exception:
                print("Could not parse title line of " + self.folder + "/" + str(self.N) + self.tag + self.appendix + str(N) + ".txt")
        except Exception:
            print("Could not read " + self.folder + "/" + str(self.N) + self.tag + self.appendix + str(N) + ".txt")


class ReadData2:
    """Parse the compact key=value format written by writeFile2.

    Each record starts with a line ``mode = <value>`` and ends just before the
    next such line (or at EOF).  Multiple runs appended to the same file are
    indexed with the ``i`` parameter (0-based).

    Attributes mirror the keys written by writeFile2:
        mode, N, f, SNR, p, loss, state, residual,
        c_p, c_s, e_in1, e_in2, e_end1, e_end2,
        bounds, optimization_method, optimization_options,
        useGPU, runtime
    ``energy`` is an alias for ``residual`` (compatibility with ReadData).
    """

    def __init__(self, filename, i=0):
        self.filename = filename
        self.i = i

        # defaults
        self.mode = ""
        self.N = 0
        self.f = 0.0
        self.SNR = 0.0
        self.p = 0.0
        self.loss = ""
        self.state = np.array([])
        self.residual = float("nan")
        self.energy = float("nan")
        self.c_p = 0.0
        self.c_s = 0.0
        self.e_in1 = np.array([0.0, 0.0, 0.0])
        self.e_in2 = np.array([0.0, 0.0, 0.0])
        self.e_end1 = np.array([0.0, 0.0, 0.0])
        self.e_end2 = np.array([0.0, 0.0, 0.0])
        self.bounds = np.array([])
        self.optimization_method = ""
        self.optimization_options = {}
        self.useGPU = False
        self.runtime = float("nan")

        with open(filename, "r") as fh:
            lines = fh.readlines()

        # Split into records at lines that start "mode = "
        record_starts = [idx for idx, ln in enumerate(lines) if ln.startswith("mode = ")]
        if i >= len(record_starts):
            raise IndexError(
                f"Record index {i} out of range — file contains {len(record_starts)} record(s)."
            )
        start = record_starts[i]
        end = record_starts[i + 1] if i + 1 < len(record_starts) else len(lines)
        record_lines = lines[start:end]

        for ln in record_lines:
            ln = ln.rstrip("\n")

            if ln.startswith("#runtime"):
                try:
                    self.runtime = float(ln.split("=", 1)[1].strip().split()[0])
                except Exception:
                    pass
                continue

            if ln.startswith("//"):
                # //optimization_options = {...}
                if "optimization_options" in ln:
                    try:
                        self.optimization_options = ast.literal_eval(
                            ln.split("=", 1)[1].strip()
                        )
                    except Exception:
                        pass
                continue

            if " = " not in ln:
                continue

            key, _, val = ln.partition(" = ")
            key = key.strip()

            if key == "mode":
                self.mode = val.strip()
            elif key == "N":
                self.N = int(val.strip())
            elif key == "f":
                self.f = float(val.strip())
            elif key == "SNR":
                self.SNR = float(val.strip())
            elif key == "p":
                self.p = float(val.strip())
            elif key == "loss":
                self.loss = val.strip()
            elif key == "state":
                self.state = np.array(ast.literal_eval(val.strip()))
            elif key == "residual":
                self.residual = float(val.strip())
                self.energy = self.residual
            elif key == "c_p":
                self.c_p = float(val.strip())
            elif key == "c_s":
                self.c_s = float(val.strip())
            elif key in ("e_in1", "e_in2", "e_end1", "e_end2"):
                # val looks like: np.array([x,y,z])
                bracket = val[val.index("["):val.rindex("]") + 1]
                setattr(self, key, np.array(ast.literal_eval(bracket)))
            elif key == "bounds":
                # val looks like: np.array([[...],[...]])
                bracket = val[val.index("["):val.rindex("]") + 1]
                self.bounds = np.array(ast.literal_eval(bracket))
            elif key == "optimization_method":
                self.optimization_method = val.strip()
            elif key == "useGPU":
                self.useGPU = val.strip().lower() == "true"

    @staticmethod
    def count(filename):
        """Return the number of records in *filename*."""
        with open(filename, "r") as fh:
            return sum(1 for ln in fh if ln.startswith("mode = "))
