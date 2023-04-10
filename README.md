# Cosmological Analysis from Angular BAO data

Likelihood for cosmological analysis of angular BAO


The files .py (Angular_BAO.py, ...) sample a markov chain assuming the flat $\Lambda$ CDM as the theoretical model, exploring the parametric space {rd*h,Om} or {rd,H0,Om}.


The Data:

     ---> Final_Table.txt: File contains the BAO measurements as described by the paper ...

          .1th Column : Redshift
          .2th Column : Angula BAO Scale measuremnts biased 
          .3th Column : Angula BAO Scale measuremnts unbiased
          .4th Column : Statistical Measuremnt's erro
          .5th Column : Systematic Measuremnt's erro
          .6th Column : Total error = sqrt(stat**2 + sys**2)         
          .7th Column : Perncentile of total error relative to the unbiased measurements 

     ---> Cov_BAO.txt : Covariance matrix between the angula BAO scale.
     
     ---> Snv.txt : Pantheon Supernova data. It can be used to combine SUpernova's likelihood with angular BAO's likelihood and get better constrains over the cosmological parameters.
     
     ---> Snvsys.txt : Matrix with the systematic erros relative Supernova measurements.
     
     
## Angular_BAO.py:


 In this analysis we use just angular BAO measurements to infer the cosmological parameters. As described in the paper ... , the expression of the likelihood is given by:



$$\chi^2_{BAO}(H_0,\Omega, r_d) = \sum_{ij} [\theta_{BAO}(z_i) - \theta_{BAO,i}]\Sigma^{-1}_{BAO,ij}[\theta_{BAO}(z_j) - \theta_{BAO,j}]$$



  Where $\theta_{BAO}(z) = \frac{r_d}{(1+z)d_A(z)}$ (with $d_A$ is the angular diameter distance).
  
  Due the dependency $\frac{r_d}{d_A}$ in the likelihood we have a degeneracy between the parameters $r_d$ and $H_0$, so the parametric space is $(r_dh,\Omega_m)$. 
  
  The priors of the free parameters $r_d, \Omega_m$.
  
## BAO_Sn_flat_prior.py:


  In this analysis we combine the likelihood with Angular BAO and Supernova. The last one is:
  
  
  $$\chi^2_{sne}(H_0,\Omega, M) = \sum_{ij} [m(z_i) - m_{i}]\Sigma^{-1}_{sne,ij}[m(z_j) - m_{j}]$$
  
  Where $M$ is the absolute magnitude of supernova, and $m(z) = 5log\left[\frac{D_L(z)}{10pc}\right] + M$
  
  
  
  
