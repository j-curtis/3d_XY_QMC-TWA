from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HamiltonianParams:
	"""Hamiltonian parameters and derived couplings for XY rotor simulations."""

	EJ: float
	EC: float
	T: float
	L: int
	M: int

	def __post_init__(self):
		if self.EC <= 0.:
			raise ValueError("EC must be positive.")
		if self.T <= 0.:
			raise ValueError("T must be positive.")
		if self.L < 1:
			raise ValueError("L must be a positive integer.")
		if self.M < 1:
			raise ValueError("M must be a positive integer.")

		object.__setattr__(self, "L", int(self.L))
		object.__setattr__(self, "M", int(self.M))

	@property
	def beta(self):
		return 1. / self.T

	@property
	def dt(self):
		return self.beta / self.M

	@property
	def Kx(self):
		return self.EJ * self.dt

	@property
	def Ky(self):
		return self.EJ * self.dt

	@property
	def Kt(self):
		# H_kin = EC * n^2 gives S_tau ~= sum (Delta theta)^2 / (4 * EC * dt).
		# Matching exp[Kt cos(Delta theta)] at small Delta theta gives this Kt.
		return 1. / (2. * self.EC * self.dt)

	@property
	def shape(self):
		return (self.L, self.L, self.M)

	@classmethod
	def from_dt(cls, EJ, EC, L, M, dt):
		M = int(M)
		T = 1. / (dt * M)
		return cls(EJ, EC, T, L, M)


@dataclass(frozen=True)
class QMCSamplingParameters:
	nburn: int
	nsample: int
	nstep: int

	def __post_init__(self):
		if self.nburn < 0:
			raise ValueError("nburn must be non-negative.")
		if self.nsample < 1:
			raise ValueError("nsample must be positive.")
		if self.nstep < 0:
			raise ValueError("nstep must be non-negative.")

		object.__setattr__(self, "nburn", int(self.nburn))
		object.__setattr__(self, "nsample", int(self.nsample))
		object.__setattr__(self, "nstep", int(self.nstep))


@dataclass(frozen=True)
class QMCSampleData:
	action_samples: np.ndarray
	OP_samples: np.ndarray
	vort_samples: np.ndarray
	theta_samples: np.ndarray
