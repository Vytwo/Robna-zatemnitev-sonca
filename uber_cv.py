from astropy.io import fits
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.optimize import curve_fit
import math


image_file = "" #slika
image_data = fits.getdata(image_file)


def doloci_soncev_radij(pot):
    with fits.open(pot) as hdul:
        data = hdul[0].data
    threshold = np.mean(data) + 0.3 * np.std(data)
    mask = data > threshold
    y_indices, x_indices = np.where(mask)
    x_center = np.mean(x_indices)
    y_center = np.mean(y_indices)
    distances = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)
    radius = np.max(distances)
    return x_center, y_center, radius

def intenziteta(cx,cy, r_max,n_bins=200):
    y, x = np.indices(image_data.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    dr = r_max / n_bins 
    radij = []
    svetlost = []
    nedolocenost = []
    for r_ in np.linspace(0,r_max, n_bins): 
        labels = np.zeros(image_data.shape, dtype=int)
        labels[(r>r_)&(r<r_+dr)]=1
        brightness = np.median(image_data[labels==1]) 
        brightness_std = np.std(image_data[labels==1]) + 0.001
        radij.append(r_+dr/2.)
        svetlost.append(brightness)
        nedolocenost.append(brightness_std)
    return radij, svetlost, nedolocenost

def md_2(radij, svetlost):
    x = np.array(radij)
    y = np.array(svetlost) / svetlost[0]
    R_konstanta = 1324


    maska_celi = x < (R_konstanta - 0.5)
    x_celi = x[maska_celi]
    y_celi = y[maska_celi]


    
    polovica_radija = np.max(x) * (100/100) ###########
    maska_polovica = x < polovica_radija
    x_fit = x[maska_polovica]
    y_fit = y[maska_polovica]


    def funck(r, tau):
        R = 1324 
        return np.exp(-tau * (1 / (np.sqrt(1 - (r / R)**2)) - 1))


    try:
        popt, pcov = curve_fit(funck, x_fit, y_fit, p0=[0.3], bounds=(0.1, 1))
        tau_opt = popt[0]
        print(f"Najboljši tau (iz prve polovice): {tau_opt:.4f}")
        print("-" * 45)
        

        y_napoved_pola = funck(x_fit, tau_opt)
        rmse_pola = np.sqrt(np.mean((y_fit - y_napoved_pola) ** 2))
        print(f"RMSE na 1/2 intervala (fit):      {rmse_pola:.4f}")
        

        y_napoved_celi = funck(x_celi, tau_opt)
        rmse_celi = np.sqrt(np.mean((y_celi - y_napoved_celi) ** 2))
        print(f"RMSE na celem intervalu:          {rmse_celi:.4f}")
        print("-" * 45)
        
    except Exception as e:
        print(f"Napaka pri prilagajanju: {e}")
        tau_opt = None


    plt.figure(figsize=(10, 6))

    plt.plot(x, y, '-', c='gray', lw=5, label='Meritev')

    #plt.axvspan(0, polovica_radija, color='blue', alpha=0.1, label='Uporabljeno za fit')

    if tau_opt is not None:

        r_plot = np.linspace(0, R_konstanta - 0.5, 500)
        y_plot = funck(r_plot, tau_opt)
        plt.plot(r_plot, y_plot, color='red', lw=2,
                label=f'Fit: tau={tau_opt:.2f} RMSE celi={rmse_celi:.3f}')

    plt.xlabel(r'$R \ [\mathrm{px}]$')
    plt.ylabel('Relativna svetlost')
    plt.xlim(0, 1500)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()
    return None

def md3(radij, svetlost):
    x = np.array(radij)
    y = np.array(svetlost) / svetlost[0]
    R_konstanta = 1320


    maska_celi = x < (R_konstanta - 0.5)
    x_celi = x[maska_celi]
    y_celi = y[maska_celi]

    polovica_radija = np.max(x) * (100/100)
    maska_polovica = x < polovica_radija
    x_fit = x[maska_polovica]
    y_fit = y[maska_polovica]


    def funck_emisija(r, tau, t_ratio):
        R = 1320
        i_ratio = t_ratio**4
        zgoraj = (i_ratio - 1) * np.exp(-tau / np.sqrt(1 - (r/R)**2)) + 1
        spodaj = (i_ratio - 1) * np.exp(-tau) + 1
        return zgoraj / spodaj


    try:
        popt, pcov = curve_fit(funck_emisija, x_fit, y_fit, 
                            p0=[0.3, 1.5], 
                            bounds=([0.01, 0.1], [10.0, 10.0]))
        
        tau_opt, t_ratio_opt = popt
        print(f"Najboljši tau (iz prve polovice): {tau_opt:.4f}")
        print(f"Najboljši t_ratio (iz prve polovice): {t_ratio_opt:.4f}")
        print("-" * 45)
        

        y_napoved_pola = funck_emisija(x_fit, tau_opt, t_ratio_opt)
        rmse_pola = np.sqrt(np.mean((y_fit - y_napoved_pola) ** 2))
        print(f"RMSE na 1/2 intervala (fit):      {rmse_pola:.4f}")
        

        y_napoved_celi = funck_emisija(x_celi, tau_opt, t_ratio_opt)
        rmse_celi = np.sqrt(np.mean((y_celi - y_napoved_celi) ** 2))
        print(f"RMSE na celem intervalu:          {rmse_celi:.4f}")
        print("-" * 45)
        
    except Exception as e:
        print(f"Napaka pri prilagajanju: {e}")
        tau_opt, t_ratio_opt = None, None

    # 4. RISANJE
    plt.figure(figsize=(10, 6))

    plt.plot(x, y, '-', c='gray', lw=5, label='Meritev')


    if tau_opt is not None:

        r_plot = np.linspace(0, R_konstanta - 0.5, 500)
        y_plot = funck_emisija(r_plot, tau_opt, t_ratio_opt)
        plt.plot(r_plot, y_plot, color='red', lw=2,
                label=f'Fit: tau={tau_opt:.2f}, $T_a/T_b$={t_ratio_opt:.2f}\nRMSE celi={rmse_celi:.3f}')
                
        
    plt.xlabel(r'$R \ [\mathrm{px}]$')
    plt.ylabel('Relativna svetlost')
    plt.xlim(0, 1500)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()
    return None

def mdu(radij, svetlost):
    def funk_linearni_model(r, *params):

        R = radij
        u = params[0] 
        
        # cos_theta = sqrt(1 - (r/R)^2)
        cos_theta = np.sqrt(np.maximum(0, 1 - (r / R) ** 2))
        
        # Model
        return 1 - u * (1 - cos_theta)


    p0 = [0.6] # 

    try:
        popt, pcov = curve_fit(funk_linearni_model, radij, svetlost, p0=p0)
        u_fit = popt[0]
        print(f"Optimalni parameter u = {u_fit:.4f}")
        

        y_model = funk_linearni_model(radij, *popt)
        rmse_lin = np.sqrt(np.mean((svetlost - y_model) ** 2))
        print(f"RMSE (linearni model) = {rmse_lin:.4f}")

    except Exception as e:
        print(f"Napaka pri fitanju: {e}")
        popt = None
    return None

def md4(radij, svetlost, n=2, show= False,R_konstanta=1320):
    x = np.array(radij)
    y = np.array(svetlost) / svetlost[0]

    fit_R = False  # STIKALO

    polovica_radija = np.max(x) * (100 / 100)
    maska_polovica = x < polovica_radija
    x_fit = x[maska_polovica]
    y_fit = y[maska_polovica]


    def funk_model4(r, *params):
        if fit_R:

            a = params[:-1]
            R = params[-1]
        else:
   
            a = params
            R = R_konstanta
            
        cos_theta = np.sqrt(np.maximum(0, 1 - (r / R) ** 2)) 
        I = 0
        for n, a_n in enumerate(a):
            I += math.factorial(n) * a_n * (cos_theta**n)
        return I


    p0 = n*[1]
    if fit_R:
        p0.append(R_konstanta)
    
    try:
        popt, pcov = curve_fit(funk_model4, x_fit, y_fit, p0=p0)
        

        print("Optimalni parametri:")
        for i in range(len(popt) - 1 if fit_R else len(popt)):
            print(f"  a{i} = {popt[i]:.4f}")
        if fit_R:
            print(f"  R_fit = {popt[-1]:.2f}")

    except Exception as e:
        print(f"Napaka pri prilagajanju: {e}")
        popt = None
    
    # 4. RISANJE
    if show:
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, "-", c="gray", lw=5, label="Meritev")
        #plt.axvspan(0, polovica_radija, color="blue", alpha=0.1, label="Uporabljeno za fit")



        if popt is not None:

            tekst_koef = "\n".join([f"a{i} = {val:.3f}" for i, val in enumerate(popt[:-1] if fit_R else popt)])
            #tekst_koef += f"\nR = {popt[-1]:.1f}"
   
            plt.annotate(f'Koeficienti:\n{tekst_koef}', 
                        xy=(0.02, 0.05),
                        xycoords='axes fraction',
                        fontsize=10, 
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)) # Okvir za boljšo berljivost


            R_final = popt[-1] if fit_R else R_konstanta
            r_plot = np.linspace(0, R_final - 0.5, 500)
            y_plot = funk_model4(r_plot, *popt)
            plt.plot(r_plot, y_plot, color='red', lw=2, label="Optimalni fit modela 4 z dvema koeficientoma")

        plt.xlabel(r"$R \ [\mathrm{px}]$")
        plt.ylabel("Relativna svetlost")
        plt.legend()
        plt.show()
    return popt

def tempstvar(popt):
    tau = np.linspace(0,2,1000)
    S = sum(a_n * (tau**n) for n, a_n in enumerate(popt))
    
    T = S**0.25
    T_1 = sum(a_n  for  a_n in popt)**0.25

    T = T * 6300. / T_1

    plt.plot(tau, T)
    plt.xlabel('$\\tau$')
    plt.ylabel('$T$ / K')
    plt.xlim(0,2);

    return tau, T

