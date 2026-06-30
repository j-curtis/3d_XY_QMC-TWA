### Code to solve 2+1 d quantum monte carlo XY model on square lattice 
### Jonathan Curtis
### 11/22/24

import numpy as np
from scipy import integrate as intg
import pickle 


### We will create a class to handle the QMC sampling 
class QMC:
	
	### This will be a useful helper method that can calculate the temperatures given a fixed trotter step size and number of steps 
	@classmethod
	def calc_temperature(cls,dt,M):
		betas = dt*M ### M may be an array of sizes
		
		return 1./betas 

	### Initialize method
	def __init__(self,EJ,EC,T,L,M):
		### this will initialize a class with the parameters for the model as well as simulation specs
		### EJ = Josephon coupling 
		### EC = Capacitive coupling
		### T = temperature 
		### L = integer physical dimension 
		### M = integer number of imaginary time steps


		### We now initialize the RNG 
		self.rng = np.random.default_rng()

		self.EJ = EJ
		self.EC = EC 
		self.T = T 

		self.L = L  
		self.M = max(M,1)

		### Now we produce the relevant array shape 
		self.shape = (self.L,self.L,self.M)

		### And the relevant time-steps 
		self.beta = 1./self.T
		self.dt = self.beta/self.M


		### Relevant coupling constants for the 3d model are 
		### This is after (1) trotterizing and (2) making Villain approximation on the time-slices 
		self.Kx = self.EJ*self.dt 
		self.Ky = self.EJ*self.dt 
		self.Kt = 1./(self.EC * self.dt) ### The coupling between neighboring time slices


		### we use an initial condition which is uniform
		self.thetas = np.zeros(self.shape)
		
		### This flag will be turned on if we want to use an over relaxation procedure 
		self.over_relaxation = False
		
		### This flag will be turned on if we want to draw from the local field distribution rather than the propose/reject form 
		self.local_field_draw = False
		

		
		### If this is set true we will print more detailed information about processes 
		self.verbose = False 
		
		### This precomputes the list of sites to loop over
		xsites = np.arange(self.L)[:,None,None]
		ysites = np.arange(self.L)[None,:,None]
		tsites = np.arange(self.M)[None,None,:]

		xsites_grid,ysites_grid,tsites_grid = np.meshgrid(xsites,ysites,tsites,indexing='ij')

		self.sites = np.stack([xsites_grid.ravel(),ysites_grid.ravel(),tsites_grid.ravel()],axis=-1)
		
		### This flag will randomize the site order that the sweep is performed in 
		self.shuffle_sites = False

	### Modifies the thetas in place one site at a time
	### Works by randomly selecting a site and performign a metropolis-hastings update
	def MCStep_random(self):

		### First we randomly propose a site 
		x = self.rng.integers(0,self.L)
		y = self.rng.integers(0,self.L)
		t = self.rng.integers(0,self.M)

		### Now we propose an update to the angle 
		delta_theta = np.pi
		new_theta = (self.thetas[x,y,t] -delta_theta + 2.*delta_theta * self.rng.random() )%(2.*np.pi)
		
		old_energy = -self.Kx*( np.cos(self.thetas[x,y,t] - self.thetas[x-1,y,t]) + np.cos(self.thetas[x,y,t] - self.thetas[(x+1)%self.L,y,t]) )
		old_energy += -self.Ky*( np.cos(self.thetas[x,y,t] - self.thetas[x,y-1,t]) + np.cos(self.thetas[x,y,t] - self.thetas[x,(y+1)%self.L,t]) )
		old_energy += -self.Kt*( np.cos(self.thetas[x,y,t] - self.thetas[x,y,t-1]) + np.cos(self.thetas[x,y,t] - self.thetas[x,y,(t+1)%self.M]) )
		
		new_energy = -self.Kx*( np.cos(new_theta - self.thetas[x-1,y,t]) + np.cos(new_theta - self.thetas[(x+1)%self.L,y,t]) )
		new_energy += -self.Ky*( np.cos(new_theta - self.thetas[x,y-1,t]) + np.cos(new_theta - self.thetas[x,(y+1)%self.L,t]) )
		new_energy += -self.Kt*( np.cos(new_theta - self.thetas[x,y,t-1]) + np.cos(new_theta - self.thetas[x,y,(t+1)%self.M]) )

		delta_E = new_energy - old_energy

		p = self.rng.random()

		if p < np.exp(-delta_E):
			self.thetas[x,y,t] = new_theta

	### This method implements a similar procedure as the local MCStep_random method but does it for a specific site = [x,y,t]
	def MCStep_site(self,site):
		x , y ,t = site[:]

		if self.local_field_draw:
			self.thetas[x,y,t] = self.draw_from_local_field(site)
		else:
			### Now we propose an update to the angle 
			delta_theta = np.pi
			new_theta = (self.thetas[x,y,t] - delta_theta + 2.*delta_theta*self.rng.random() )%(2.*np.pi)
			
			delta_E = self.local_action(new_theta,site) - self.local_action(self.thetas[x,y,t],site)

			p = self.rng.random()

			if p < np.exp(-delta_E):
				self.thetas[x,y,t] = new_theta

	### This method performs an entire sweep over the lattice of MCStep_site method 
	def MCSweep(self):
		indices = np.arange(self.sites.shape[0])
		
		if self.shuffle_sites: indices = self.rng.permutation(indices)
		 
		for i in indices:
			self.MCStep_site(self.sites[i,:])
			
		if self.over_relaxation:
			self.over_relaxation_sweep()


	### Local energy function is useful for calling in MC step updates 
	### theta_val is the value of the angle at size x,y,t
	### it is not assumed to be the value stored in the configuraiton so that this can be used to also evaluate proposed energy
	def local_action(self,theta_val,site):
		x,y,t = site[:]
		xterms = -self.Kx*( np.cos(theta_val - self.thetas[x-1,y,t]) + np.cos(theta_val - self.thetas[(x+1)%self.L,y,t]) )
		yterms = -self.Ky*( np.cos(theta_val - self.thetas[x,y-1,t]) + np.cos(theta_val - self.thetas[x,(y+1)%self.L,t]) )
		tterms = -self.Kt*( np.cos(theta_val - self.thetas[x,y,t-1]) + np.cos(theta_val - self.thetas[x,y,(t+1)%self.M]) )

		return xterms+yterms+tterms
		
	### This computes the local self-consistent field for use with the heat-bath Gibbs method and draws a new angle according to this 
	### Return new theta 
	def draw_from_local_field(self,site):
		x,y,t = site[:] 
	
		theta_out = None
	
		psi =  self.Kx*( np.exp(-1.j*self.thetas[x-1,y,t]) + np.exp(-1.j*self.thetas[(x+1)%self.L,y,t]) )
		psi += self.Ky*( np.exp(-1.j*self.thetas[x,y-1,t]) + np.exp(-1.j*self.thetas[x,(y+1)%self.L,t]) )
		psi += self.Kt*( np.exp(-1.j*self.thetas[x,y,t-1]) + np.exp(-1.j*self.thetas[x,y,(t+1)%self.M]) )
		
		kappa = float(np.abs(psi))
		
		if kappa < 1e-14: ### Safeguard for perfect cancellation 
			theta_out = self.rng.uniform(-np.pi, np.pi)
		else:
			mu = float(np.angle(psi))
			# numpy's vonmises returns in [-pi, pi]; perfect for us
			theta_new = self.rng.vonmises(mu=mu, kappa=kappa)
			
		return theta_new
		

	### This method computes the local self-consistent field on a given site 
	def get_local_field(self,site):
		x,y,t = site[:]

		xterms = -self.Kx*( np.exp(1.j*self.thetas[x-1,y,t]) + np.exp(1.j*self.thetas[(x+1)%self.L,y,t]) )
		yterms = -self.Ky*( np.exp(1.j*self.thetas[x,y-1,t]) + np.exp(1.j*self.thetas[x,(y+1)%self.L,t]) )
		tterms = -self.Kt*( np.exp(1.j*self.thetas[x,y,t-1]) + np.exp(1.j*self.thetas[x,y,(t+1)%self.M]) )

		return xterms+yterms+tterms
	
	### This implements an over-relaxation sweep 
	def over_relaxation_sweep(self):
	
		indices = np.arange(self.sites.shape[0])
		
		if self.shuffle_sites: indices = self.rng.permutation(indices)
		 
		for i in indices:
			site = self.sites[i,:]
			x,y,t = site[:]
			local_field = self.get_local_field(site)
			
			if np.abs(local_field) != 0.:
				local_field = local_field/np.abs(local_field) ### normalize to unit phasor
				
				### convert to cartesian vector 
				nx = np.real(local_field)
				ny = np.imag(local_field) 
				
				### get the current local theta 
				theta = self.thetas[x,y,t]
				
				### Now we reflect this 
				sx = np.cos(theta)
				sy = np.sin(theta)
				
				proj = nx*sx + ny*sy 
				sx_new = 2.*proj*nx - sx 
				sy_new = 2.*proj*ny - sy 
				
				self.thetas[x,y,t] =  np.arctan2(sy_new, sx_new) 
				
	
	##########################
	### SAMPLE OBSERVABLES ###
	##########################

	### This method computes the total free energy density for a particular configuration
	def get_action(self,thetas):
		### This generates a list of nn indices to roll arrays by
		nn_indices = [(1,0,0),(0,1,0),(0,0,1)]
		### For each nearest neighbor this is the corresponding spin stiffness in that direction 
		nn_Ks = [ self.Kx,self.Ky,self.Kt ]

		action = 0.

		for i in range(3):
			K = nn_Ks[i]
			dthetas = thetas - np.roll(thetas,nn_indices[i],axis=(0,1,2))

			action += - np.sum( K*np.cos(dthetas) )

		return action

	@classmethod
	def angle_diff(cls,theta1,theta2):
		### returns a properly modded angular difference for computing vorticity
		return -np.pi + (np.pi + theta1-theta2)%(2.*np.pi)

	### This computes the vorticity distribution for a given set of angles 
	@classmethod
	def get_vorticity(cls,thetas):
		### This generates a list of nn indices to roll arrays by
		### Note we index the rolls absolutely with respect to the origin of the first array
		nn_indices = [(-1,0,0),(-1,-1,0),(0,-1,0),(0,0,0)]

		vorticity = np.zeros_like(thetas)
		
		for i in range(len(nn_indices)):
			indx1 = nn_indices[i]
			indx2 = nn_indices[i-1]
			vorticity += cls.angle_diff( np.roll(thetas,indx1,axis=[0,1,2]) , np.roll(thetas,indx2,axis=[0,1,2]) )

		return vorticity

	### This method computes the mean order parameter as < e^{itheta} > averaged over space and imaginary time
	@classmethod
	def get_OP(cls,thetas):
		return np.mean(np.exp(1.j*thetas))
		
	###########################
	### MC SAMPLING METHODS ###
	###########################


	### Sets the parameters for sampling 
	def set_sampling(self,nburn,nsample,nstep):
	
		### Parameters relevant for MC sampling 
		self.nburn = nburn  ### Number of burn steps. Should be updated at some point to a converging algorithm which runs until converged 
		self.nsample = nsample ### How many samples we want
		self.nstep = nstep ### How many steps between each sample 

		self.theta_samples = np.zeros((self.L,self.L,self.M,self.nsample))
		self.action_samples = np.zeros(self.nsample)
		self.vort_samples = np.zeros((self.L,self.L,self.M,self.nsample))
		self.OP_samples = np.zeros(self.nsample,dtype=complex)

	### This method implements the burn loop using the single MCStep method for nburn iterations 
	def burn(self):
		if self.verbose: print("Begin burn loop.")
		for i in range(self.nburn):
			
			self.MCSweep()

	### We now generate samples and sample the free energy density 
	def sample(self):
		counter = 0 
		while counter < self.nsample:
			### Record the sample
			self.theta_samples[...,counter] = self.thetas
			self.action_samples[counter] = self.get_action(self.thetas)
			self.vort_samples[...,counter] = self.get_vorticity(self.thetas)
			self.OP_samples[counter] = self.get_OP(self.thetas)

			### Now we run for a number of steps 
			for i in range(self.nstep):
				self.MCSweep()

			### Update the counter 
			counter += 1

	### This will start a hot start with a precomputed sample configuration 
	def hot_start(self,sample_filename):
		### We pass a filename where the data is stored 
		with open(sample_filename,'rb') as f:
			sample = pickle.load(f) 
			self.thetas = sample
		
		

### Compatibility with demler_tools
def run_MC_sims(save_filename,Ej,Ec,L,M,dt,nburn,nsample,nstep,hot_start_filename=None):
	M = int(M)
	T = QMC.calc_temperature(dt,M)
	sim = QMC(Ej,Ec,T,L,M)
	sim.over_relax = True
	sim.local_field_draw = False #True 
	sim.shuffle_sites = True 
	sim.set_sampling(nburn,nsample,nstep)


	### Now we handle the case of a hot start 
	if hot_start_filename is not None:
		sim.hot_start(hot_start_filename)
		
	sim.burn()
	sim.sample()

	with open(save_filename, 'wb') as out_file:
        	pickle.dump((sim.action_samples,sim.OP_samples,sim.vort_samples,sim.theta_samples), out_file)
        	
        ### now some smaller suffixed files
        ### action samples 
	with open(save_filename+"_action_samples",'wb') as out_file:
		pickle.dump(sim.action_samples, out_file)
        
        ### OP samples 
	with open(save_filename+"_OP_samples",'wb') as out_file:
		pickle.dump(sim.OP_samples,out_file)




