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

path = './data'


########################################################################################################################################################################
#_______________________________________________________________________________________________________________________________________________________________________
#                                                                          Defining Functions
#
########################################################################################################################################################################


class Bayes_Angular_BAO:
    def __init__(self, c, redshift_bao,redshift_sn, BAO,mb, Cov_bao,C_tot_sn, invCOV_bao,invCOV_sn, error_bao, x0):

        self.c = c
        self.redshift_bao = redshift_bao
        self.redshift_sn = redshift_sn
        self.BAO = BAO
        self.mb = mb
        self.Cov_bao = Cov_bao
        self.C_tot_sn = C_tot_sn
        self.invCOV_bao = invCOV_bao   
        self.invCOV_sn = invCOV_sn
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

    def Function(self,z,Om):
        c100 = self.c/100
        I = []
        for i in range(len(z)):
            I.append(integrate.quad(lambda x: 1/self.E(Om,x), 0,z[i])[0])
        I = np.array(I)
        return 5*np.log10((1+z)*c100*I)+25

    def Function_mb(self,z,Om, Mb):
        return self.Function(z,Om) + Mb 

    def lnlike_bao(self,x):
        rh,Om, Mb = x
        Fun_bao = self.Angle_model(self.redshift_bao,rh,Om)
        delta_bao = self.BAO - Fun_bao    
        #Fun_sn = self.Function(self.redshift_sn,Om)
        #delta_sn = mb -Fun_sn - Mb
        return  -0.5*np.dot(delta_bao,np.dot(self.invCOV_bao,delta_bao))#-0.5*np.sum((delta_bao/error_bao)**2)


    def lnlike_sn(self,x):
        rh,Om, Mb = x
        #Fun_bao = self.Angle_model(self.redshift_bao,rh,Om)    
        Fun_sn = self.Function(self.redshift_sn,Om)
        delta_sn = self.mb -Fun_sn - Mb
        #delta_bao = BAO - Fun_bao
        return -0.5*np.dot(delta_sn,np.dot(self.invCOV_sn,delta_sn))


    def lnlike(self,x):
        rh,Om, Mb = x
        Fun_bao = self.Angle_model(self.redshift_bao,rh,Om)    
        Fun_sn = self.Function(self.redshift_sn,Om)
        delta_sn = self.mb -Fun_sn - Mb
        delta_bao = self.BAO - Fun_bao
        return -0.5*np.dot(delta_sn,np.dot(self.invCOV_sn,delta_sn)) -0.5*np.dot(delta_bao,np.dot(self.invCOV_bao,delta_bao)) #-0.5*np.sum((delta_bao/error_bao)**2)


    def lnprior(self,x):
        rh,Om, Mb = x  
        delta_mb = Mb -x0[2]
        Om_min = 0.0
        Om_max = 1.0
        Mb_min = x0[2] - 10.0
        Mb_max = x0[2] + 10.0
        rh_min = x0[0] - 20.0
        rh_max = x0[0] + 20.0
        if ( (Om_min<Om<Om_max) and (rh_min<rh<rh_max) and (Mb_min<Mb<Mb_max) ):
            return 0.0        
        return -np.inf


    def lnprob(self,x):
        lp = self.lnprior(x)
        if not np.isfinite(lp):
            return -np.inf
        #print(x)
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
Cov_bao = np.loadtxt(path+'/Cov_BAO.txt')
invCOV_bao = np.linalg.inv(Cov_bao)



#                                                 ###########################  Supernovae Data  #############################

Snv_data = np.loadtxt(path+'/Snv.txt', usecols = (1,4,5))
redshift_sn = Snv_data[:,0]
mb = Snv_data[:,1] 
Snv_sys_data = np.loadtxt(path+'/Snvsys.txt')
Csys = np.reshape(Snv_sys_data[1:], (1048,1048))
A = np.zeros([1048,1048])

for i in range(0,1048):
    A[i,i] = Snv_data[i,2]**2

C_tot_sn = A+Csys
invCOV_sn = np.linalg.inv(C_tot_sn)




########################################################################################################################################################################
#_______________________________________________________________________________________________________________________________________________________________________
#                                                                           Building the Chain
#
########################################################################################################################################################################


c = 299792.458
x0 = [100, 0.32, -18.57]
#c, redshift_bao,redshift_sn, BAO,mb, Cov,C_tot_sn, invCOV,invCOV_sn, error_bao, x0
instan_Bayes = Bayes_Angular_BAO(c,redshift_bao,redshift_sn, BAO,mb, Cov_bao,C_tot_sn, invCOV_bao,invCOV_sn, error_bao, x0) 
ndim, nwalkers = 3, 10
Transition = lambda x: [np.random.normal(x[0], 0.1), np.random.normal(x[1], 0.1), np.random.normal(x[2], 0.0001)]
pos = [np.array(Transition(x0)) for i in range(nwalkers)]
filename = 'BAO+Snv_flat.h5' 
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

#class emcee:

sampler = emcee.EnsembleSampler(nwalkers, ndim, instan_Bayes.lnprob, backend=backend)

#run the MCMC:

steps =1000#10000
sampler.run_mcmc(pos, steps, progress = True)

########################################################################################################################################################################
#_______________________________________________________________________________________________________________________________________________________________________
#                                                                      Ploting the Posterior Distributions
#
########################################################################################################################################################################

burn = 100#1000
reader = emcee.backends.HDFBackend('BAO+Snv_flat.h5')
names = [r'r_dh', r'\Omega_m',r'\hat{M}_b']
labels =  names
setting = {'range_ND_contour':1}
SampleParamet = MCSamples(samples=np.transpose(reader.get_chain(thin=1)[burn:,:,:], (1,0,2)),names = names, labels = labels, ranges = {r'r_dh':(x0[0] - 20.0,x0[0] + 20.0), r'\Omega_m': (0.0, 1.0), r'\hat{M}_b': (x0[2]-10., x0[2]+10.) }, settings = setting)
stats = SampleParamet.getMargeStats()
densi_A = SampleParamet.get1DDensityGridData("r_dh")
densi_B = SampleParamet.get1DDensityGridData("\Omega_m")
densi_C = SampleParamet.get1DDensityGridData(r'\hat{M}_b')


DENSI = [densi_A, densi_B, densi_C]
Mode = []

for densi in DENSI: 
    para = densi.x[densi.P ==1][0]
    Mode.append(para)


print('Mode = ', Mode)


limit_lower0 = []
limit_upper0 = []
limit_lower1 = []
limit_upper1 = []

for i in range(len(names)):

    par = stats.parWithName(names[i])
    #Best_values.append(Mode[i])
    limit_lower0.append(Mode[i] -par.limits[0].lower)
    limit_upper0.append(par.limits[0].upper-Mode[i])
    limit_lower1.append(Mode[i] -par.limits[1].lower)
    limit_upper1.append(par.limits[1].upper-Mode[i])
    print('\n\n')


Best_values = Mode

np.savetxt(f'Best_values_BAO+Snv_flat.txt', Best_values)
np.savetxt(f'lower_BAO+Snv_flat.txt', (limit_lower0, limit_lower1))
np.savetxt(f'upper_BAO+Snv_flat.txt', (limit_upper0, limit_upper1))



e = plots.get_subplot_plotter()
e.triangle_plot(SampleParamet, filled=True,title_limit=0,contour_colors= ['xkcd:turquoise'],legend_loc='upper right', markers = {'r_dh':Mode[0], '\Omega_m':Mode[1], r'\hat{M}_b':Mode[2]})
e.export('triplot_BAO+Sn_flat.pdf')






