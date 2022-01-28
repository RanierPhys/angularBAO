import numpy as np
import scipy.integrate as integrate
import matplotlib.pylab as plt
import emcee
from numba import jit, njit
from numba import f8
from getdist import plots, MCSamples
import getdist
import emcee
from scipy.optimize import dual_annealing
from scipy.special import gamma
from scipy.optimize import curve_fit
from scipy import linalg

path = './data'

########################################################################################################################################################################
#_______________________________________________________________________________________________________________________________________________________________________
#                                                                          Defining Functions
#
########################################################################################################################################################################




class Bayes_Angular_BAO:
    def __init__(self, c, redshift_bao, BAO, Cov, invCOV, error_bao, x0):

        self.c = c
        self.redshift_bao = redshift_bao
        self.BAO = BAO
        self.Cov = Cov
        self.invCOV = invCOV   
        self.error_bao = error_bao
        self.x0 = x0

    def E(self,Om,z):
        return (Om*((1+z)**3)+1-Om)**0.5

    def Angle_model(self,z,rh,Om):
        c = self.c
        A = 100*rh/c
        I = []
        for i in range(len(z)): 
            I.append(integrate.quad(lambda x: 1/self.E(Om,x), 0,z[i])[0])   
        I = np.array(I)
        return (180/np.pi)*(A/I)

    def lnlike(self,x):
        rh,Om = x
        Fun_bao = self.Angle_model(self.redshift_bao,rh,Om)       
        delta_bao = self.BAO - Fun_bao
        return -0.5*np.dot(delta_bao,np.dot(self.invCOV,delta_bao))
    


    def lnprior(self,x):
        rh,Om = x 
        Om_min = 0.0
        Om_max = 1
        rh_min = self.x0[0] - 20.0
        rh_max = self.x0[0] + 20.0
        if ( (Om_min<Om<Om_max) and (rh_min<rh<rh_max) ):
            return 0.0        
        return -np.inf


    def lnprob(self,x):
        lp = self.lnprior(x)
        if not np.isfinite(lp):
            return -np.inf
    
        return lp + self.lnlike(x)


########################################################################################################################################################################
#_______________________________________________________________________________________________________________________________________________________________________
#                                                                          Open the Data
#
########################################################################################################################################################################



#                                                 ###########################  Angular BAO Data  #############################

Total_Data = np.loadtxt(path+'/Final_Table.txt')
BAO = Total_Data[:,2]
redshift_bao = Total_Data[:,0]
error_bao = Total_Data[:,5]
Cov = np.loadtxt(path+'/Cov_BAO.txt')
invCOV = np.linalg.inv(Cov)


########################################################################################################################################################################
#_______________________________________________________________________________________________________________________________________________________________________
#                                                                           Building the Chain
#
########################################################################################################################################################################


c = 299792.458
x0 = [100, 0.32]
instan_Bayes = Bayes_Angular_BAO(c,redshift_bao, BAO, Cov, invCOV, error_bao, x0 ) 
ndim, nwalkers = 2, 10
Transition = lambda x: [np.random.normal(x[0], 0.1), np.random.normal(x[1], 0.01)]
pos = [np.array(Transition(x0)) for i in range(nwalkers)]
filename = 'Angular_BAO.h5' 
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

#class emcee:

sampler = emcee.EnsembleSampler(nwalkers, ndim, instan_Bayes.lnprob, backend=backend)

#run the MCMC:

steps =1000# for the paper was used 100000
sampler.run_mcmc(pos, steps, progress = True)

########################################################################################################################################################################
#_______________________________________________________________________________________________________________________________________________________________________
#                                                                      Ploting the Posterior Distributions
#
########################################################################################################################################################################

burn = 100#for the paper was used 10000
reader = emcee.backends.HDFBackend('Angular_BAO.h5')
names = [r'r_dh', r'\Omega_m']
labels =  names 
setting = {'range_ND_contour':1}
SampleParamet = MCSamples(samples=np.transpose(reader.get_chain(thin=1)[burn:,:,:], (1,0,2)),names = names, labels = labels, ranges = {r'r_dh':(x0[0] - 20.0,x0[0] + 20.0), r'\Omega_m': (0.0, 1.0) }, settings = setting)
stats = SampleParamet.getMargeStats()
densi_A = SampleParamet.get1DDensityGridData("r_dh")
densi_B = SampleParamet.get1DDensityGridData("\Omega_m")
DENSI = [densi_A, densi_B]
Mode = []

for densi in DENSI: 
    para = densi.x[densi.P ==1][0]
    Mode.append(para)
print('Mode = ', Mode)

Best_values = []
limit_lower0 = []
limit_upper0 = []
limit_lower1 = []
limit_upper1 = []

for i in range(len(names)):

    par = stats.parWithName(names[i])
    limit_lower0.append(Mode[i] -par.limits[0].lower)
    limit_upper0.append(par.limits[0].upper-Mode[i])
    limit_lower1.append(Mode[i] -par.limits[1].lower)
    limit_upper1.append(par.limits[1].upper-Mode[i])
    print('\n\n')

np.savetxt(f'Best_values_BAO_only.txt', Mode)
np.savetxt(f'lower_BAO_only.txt', (limit_lower0, limit_lower1))
np.savetxt(f'upper_BAO_only.txt', (limit_upper0, limit_upper1))


e = plots.get_subplot_plotter()
e.triangle_plot(SampleParamet, filled=True,contour_colors= ['xkcd:turquoise'],legend_loc='upper right', markers = {'r_dh':Mode[0], '\Omega_m':Mode[1]})
e.export('Angular_BAO.pdf')
