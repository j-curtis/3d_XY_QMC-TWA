### Code to solve 2+1 d quantum monte carlo XY model on square lattice 
### Jonathan Curtis
### 11/22/24

import numpy as np
from matplotlib import pyplot as plt 
import time 


### We will create a class to handle the simulations 
class QMC:
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
		#self.thetas = self.rng.random(size =self.shape)*2.*np.pi

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

		### Now we propose an update to the angle 
		delta_theta = np.pi
		new_theta = (self.thetas[x,y,t] - delta_theta + 2.*delta_theta*self.rng.random() )%(2.*np.pi)
		
		delta_E = self.local_action(new_theta,site) - self.local_action(self.thetas[x,y,t],site)

		p = self.rng.random()

		if p < np.exp(-delta_E):
			self.thetas[x,y,t] = new_theta

	### This method performs an entire sweep over the lattice of MCStep_site method 
	def MCSweep(self):
		### From ChatGPT
		xsites = np.arange(self.L)[:,None,None]
		ysites = np.arange(self.L)[None,:,None]
		tsites = np.arange(self.M)[None,None,:]

		xsites_grid,ysites_grid,tsites_grid = np.meshgrid(xsites,ysites,tsites,indexing='ij')

		sites = np.stack([xsites_grid.ravel(),ysites_grid.ravel(),tsites_grid.ravel()],axis=-1)

		for i in range(sites.shape[0]):
			self.MCStep_site(sites[i,:])

	### Local energy function is useful for calling in MC step updates 
	### theta_val is the value of the angle at size x,y,t
	### it is not assumed to be the value stored in the configuraiton so that this can be used to also evaluate proposed energy
	def local_action(self,theta_val,site):
		x,y,t = site[:]
		xterms = -self.Kx*( np.cos(theta_val - self.thetas[x-1,y,t]) + np.cos(theta_val - self.thetas[(x+1)%self.L,y,t]) )
		yterms = -self.Ky*( np.cos(theta_val - self.thetas[x,y-1,t]) + np.cos(theta_val - self.thetas[x,(y+1)%self.L,t]) )
		tterms = -self.Kt*( np.cos(theta_val - self.thetas[x,y,t-1]) + np.cos(theta_val - self.thetas[x,y,(t+1)%self.M]) )

		return xterms+yterms+tterms

	### Modifies the thetas in place 
	### Does an even/odd sublattice update in parallel
	def MCStep_checkerboard(self):
		### This implements one time step of the Metropolis update
		

		### TAKEN FROM CHAT GPT
		########################
		
		### we define a function for updating an entire sublattice 
		def sl_update(mask): 

			xs, ys, ts = np.where(mask)

			for idx in np.random.permutation(len(xs)):
				x, y, t = xs[idx], ys[idx], ts[idx]
				old_theta = self.thetas[x, y, t]
				old_E = self.local_energy(old_theta, x, y, t)

				delta_theta = np.pi/3.
				new_theta = (old_theta + np.random.uniform(-delta_theta, delta_theta)) % (2.*np.pi)

				new_E = self.local_energy(new_theta, x, y, t)
				dE = new_E - old_E

				if self.rng.random() < np.exp(-dE):
					self.thetas[x,y,t] = new_theta

		########################
		sl_update(self.even_mask)
		sl_update(self.odd_mask)



		# ### We implement by first breaking in to even and odd sublattices
		# ### Each of these can be updated independently of the other 
		# ### We index the sublattice by s = 0,1 for odd or even
		# for s in range(2):
		# 	### We compute the self-consistent fields for all sites 
		# 	### Being careful about roll
		# 	nn_indices = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1) ]
		# 	nn_Ks = [ self.Kx,self.Kx,self.Ky,self.Ky,self.Kt,self.Kt ]
		# 	### For each nearest neighbor this is the corresponding spin stiffness in that direction 

		# 	#self_consistent_field = sum([ nn_Ks[nn]*np.exp(-1.j*np.roll(thetas,nn_indices[nn])) for nn in range(len(nn_indices)) ])
		# 	self_consistent_field = sum([ nn_Ks[nn]*np.exp(-1.j*np.roll(thetas, shift=nn_indices[nn], axis=(0,1,2) ) ) for nn in range(len(nn_indices)) ]) 

		# 	new_thetas = self.rng.random(self.shape)*2.*np.pi 
		# 	### These will be the proposal angles for both odd and even, we only need to update SCF after first sweep
		# 	delta_Es = -np.real( ( np.exp(1.j*new_thetas) - np.exp(1.j*thetas) )*self_consistent_field )

		# 	### Now we form an array of accept probabilities
		# 	thresholds = np.exp(-delta_Es)

		# 	### We generate an array of random floats in [0,1] to compare 
		# 	probs = self.rng.random(size=self.shape)

		# 	### This will be one if the entry for that site is accepted and 0 else
		# 	accepts = (probs < thresholds).astype(float)

		# 	### We now elementwise replace the old angles with those that should be updated 
		# 	### We use x ->  x'' = (1-p)*x + p*x' where p is 0,1 depending on whether we accept x' over x (p = 1 is accept x')
		# 	### We mask only the even sublattice and update it 
		# 	### We generate an array mask 
		# 	### sublattice A is if x + y + t is even 
		# 	### This should generate array mask for sublattice A 
		# 	### This piece courtesy of chatGPT
		# 	x = np.arange(self.L)[:,None,None]
		# 	y = np.arange(self.L)[None,:,None]
		# 	t = np.arange(self.M)[None,None,:]
		# 	mask_SL_A = (x+y+t)%2 == 0 

		# 	mask = mask_SL_A if s == 0 else ~mask_SL_A ### for sublattice B we flip the mask
		# 	thetas[mask] += accepts[mask] * (new_thetas[mask] - thetas[mask])	

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
		### We want A_v = [ sin(theta_{r+x} - theta_r) + sin(theta_{r+x+y} - theta_{r+x} ) + sin(theta_{r+y}-theta_{r+x+y}) + sin(theta_r - theta_{r+y}) ]/4 
		#nn_indices = [(-1,0,0),(-1,1,0),(0,-1,0),(0,0,0)]
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





def main():

	dataDir = "../data/07102025_1/"


	EJ = 1.
	EC = 0.05
	nTs = 15
	Ts = np.linspace(0.5,3.,nTs)
	L = 40
	M = 1

	nburn = 10000
	nsample = 100
	nstep = 500

	actions = np.zeros((nTs,nsample))
	OPs = np.zeros((nTs,nsample),dtype=complex)
	angles = np.zeros((nTs,L,L,M,nsample))
	vorts = np.zeros((nTs,L,L,M,nsample))
	
	t0 = time.time()
	
	for i in range(nTs):
	
		tloop1 = time.time()
		sim = QMC(EJ,EC,Ts[i],L,M)
		sim.set_sampling(nburn,nsample,nstep)

		sim.burn()
		sim.sample()

		actions[i,:] = sim.action_samples
		OPs[i,:] = sim.OP_samples
		angles[i,...] = sim.theta_samples
		vorts[i,...] = sim.vort_samples 
		
		tloop2 = time.time()
		print(str(i+1)+"/"+str(nTs)+": ",tloop2 - tloop1,"s")

	t1 = time.time()
	
	np.save(dataDir+"actions.npy",actions)
	np.save(dataDir+"OPs.npy",OPs)
	np.save(dataDir+"angles.npy",angles)
	np.save(dataDir+"vorts.npy",vorts)
	np.save(dataDir+"Ts.npy",Ts)
	
	
	try:
		with open(dataDir+"meta.txt",'w') as file:
			file.write("EJ: {ej}\n".format(ej=EJ))
			file.write("EC: {ec}\n".format(ec=EC))
			file.write("L: {l}\n".format(l=L))
			file.write("M: {m}\n".format(m=M))
			file.write("nburn: {nb}\n".format(nb=nburn))
			file.write("nsample: {ns}\n".format(ns=nsample))
			file.write("nstep: {ns}\n".format(ns=nstep))
			file.write("total time: {t}s\n".format(t = t1-t0))


	except Exception as e:
		print(f"An error occurred: {e}")


if __name__ == "__main__":
	main()











