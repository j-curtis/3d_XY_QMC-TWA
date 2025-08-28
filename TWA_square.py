### Code to solve semiclassical EOM for 2D XY model sampled on a square lattice
### Jonathan Curtis
### 08/28/25

import numpy as np
from scipy import integrate as intg
from matplotlib import pyplot as plt 
import time 
import pickle 
import QMC_square as qmc



### This class operates on the output of the QMC sampler and implements the real time dynamics 
class TWDynamics:
	"""Accepts a QMC sample class and implements the real time dynamics according to truncated Wigner approximation"""
	def __init__(self,qmc_sample,tf,ntimes):
		self.qmc = qmc_sample ### Samples is an array of the shape (L,L,nsample) and is a set of samples of instantenous LxL snapshots of thetas 
		### We extract the samples 
		self.samples = self.qmc.theta_samples[:,:,0,:] ### We project to just the m=0 slice of the LxLxM array 
		
		self.shape = self.samples.shape
		
		self.nsamples = self.shape[-1]
		self.L = self.shape[0]
		
		### Simulation time parameters 
		self.tf = tf 
		self.ntimes = ntimes 
		self.times = np.linspace(0.,self.tf,self.ntimes) 
		
		self.sim_shape = (self.ntimes,2,*self.shape) ### Simulation shapes have an extra axis which is first by output from the ODE_solve method as well as an extra axis for velocity dof
	
	########################
	### INTERNAL METHODS ###
	########################
	
	### This method will be the RHS of the EOM 
	def _eom_rhs(self,t,X):
		### First we reshape the dof 
		### We encode the coordinates in the first half of the array and the velocities in the second half 
		dof = X.reshape((2,*self.shape) )
		thetas = dof[0,...]
		theta_dots = dof[1,...]
		
		dXdt = np.zeros_like(dof)
		
		dXdt[0,...] = theta_dots ### Update thetas according to the velocities
		
		nns = [ [1,0,0],[-1,0,0],[0,1,0],[0,-1,0] ] ### Amount to roll by on each axis to form the spatial couplings  
		dXdt[1,...] = -self.qmc.EC*self.qmc.EJ*sum([ np.sin(thetas - np.roll(thetas,nn,[0,1,2]))  for nn in nns ]) 
		
		### Finally we reflatten and send out 
		return dXdt.ravel()	
		
	######################
	### SET PARAMETERS ###
	######################	
		
	### Allows to modify simulation parameters 
	def set_simulation_times(self,tf,ntimes):
		self.tf = tf 
		self.ntimes = ntimes 
		self.times = np.linspace(0.,self.tf,self.ntimes)
		
		
	###################
	### RUN METHODS ###
	###################
	
	### This method will run the EOM 
	def run_dynamics(self):
		### First we generate the initial conditions 
		### For the time we will sample the initial velocities to be zero 
		### We now ravel these together with the samples in to the initial conditions
		
		### Pack initial conditions of angles in first half and velocities (taken to be zero) in second half
		X0 = np.stack([self.samples,np.zeros_like(self.samples) ]) 

		### Now we flatten the array 
		X0 = X0.ravel()

		sol = intg.solve_ivp(self._eom_rhs,(0.,self.tf),X0,t_eval=self.times) 
		
		### We reshape into (ntimes,2,L,L,nsamples)
		self.dof_t = (sol.y.T).reshape(self.sim_shape) ### First we bring the time axis first, then we reshape
		self.trajectories = self.dof_t[:,0,...] ### We also cut out the velocities as we usually don't care about them 







	
### Compatibility with demler_tools
def run_TWA_sims(save_filename,Ej,Ec,T,L,M,nburn,nsample,nstep,tf,ntimes,over_relax=False):

	sim = qmc.QMC(Ej,Ec,T,L,M)
	sim.over_relax = over_relax
	sim.set_sampling(nburn,nsample,nstep)
	
	sim.burn()
	sim.sample()
	
	twa = TWDynamics(sim,tf,ntimes)
	twa.run_dynamics()
	
	### Now we generate TW dynamical trajectories

	with open(save_filename, 'wb') as out_file:
        	pickle.dump((twa.dof_t), out_file)





