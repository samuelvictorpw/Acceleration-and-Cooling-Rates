import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as spi
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.special import kv 
import warnings 
import time 
import csv 
from scipy.integrate import solve_ivp
import os
from joblib import Parallel, delayed
from tqdm import tqdm

# (a criação da pasta de saída agora é feita por execução, dentro de run_pipeline)

# Silencia avisos numéricos que são esperados nos extremos das integrais
warnings.filterwarnings("ignore", category=spi.IntegrationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Constantes Físicas
c = 2.998e10                   
e = 4.803e-10                  
m_e = 9.109e-28                
m_p = 1.6726e-24               
h = 6.626e-27                  
gev_to_erg = 1.60218e-3       
eV_to_erg = 1.60218e-12       
mec2_erg = m_e * c**2          
sigma_T = 6.65e-25             
G = 6.674e-8
r_e = 2.8179403227e-13
Z_0_rg = 50                    

# Funções Básicas de Geometria e Cinemática
def jet_radius(Z_pos, zetaR):
    return Z_pos * np.tan(zetaR)

def calculate_B0(L_j, r_j_val, v_b):
    return np.sqrt((8 * L_j) / (v_b * (r_j_val)**2))

def calculate_B_field(B_0_val, Z_pos, Z_launch, m):
    if Z_pos <= 0: 
        return 0.0
    return B_0_val * (Z_launch / Z_pos)**m

def calculate_z_acc(z_0_rg, q_m, m):
    zacc = z_0_rg * q_m**(1/(2-2*m))
    return zacc

def calculate_B_prime(B_val, Gamma_b):
    return np.sqrt(3 / (2 * Gamma_b**2 + 1)) * B_val

def calculate_E_prime(Gamma_b, E_obs, beta, mass, c, thetaR):
    E_rest = mass * c**2
    E_obs_safe = np.maximum(E_obs, E_rest) 
    return Gamma_b * (E_obs_safe - beta * np.cos(thetaR) * np.sqrt(E_obs_safe**2 - E_rest**2))

def t_acc_rate(n, c, e, B_prime, E_prime):
    return (n * c * e * B_prime) / E_prime

def t_sync_rate(m_e, B_prime, E_prime, c, sigma_T):
    return (4.0/3.0) * (sigma_T * B_prime**2 * E_prime) / (m_e**2 * c**3 * 8 * np.pi)

def t_adiabatic_rate(Z_pos, beta, c):
    v_bulk = beta * c
    return (2.0/3.0) * (v_bulk / Z_pos)

def calculate_nc(z, L_j, Gamma_b, xi_j_deg, q_rel):
    xi_j = np.deg2rad(xi_j_deg)
    beta_b = np.sqrt(1 - 1/Gamma_b**2)
    v_b = beta_b * c
    m_dot_j = L_j / (c**2 * (Gamma_b - 1))
    numerator = (1 - q_rel) * m_dot_j
    denominator = m_p * np.pi * (z**2) * (np.tan(xi_j)**2) * v_b
    return numerator / denominator

def t_bremss_rate(E_prime_erg, n_c_prime, c):
    alpha = 1.0/137.0  
    gamma = E_prime_erg / (m_e * c**2)
    rate = 4 * alpha * (3.0 * sigma_T / (8.0 * np.pi)) * n_c_prime * c * (np.log(2*gamma) - 1.0/3.0)
    return rate

def calculate_Le(L_j, q_rel, a):
    return (L_j * q_rel) / (1.0 + a)

def calculate_Delta_V(r_j_cm, delta_z_cm):
    return np.pi * (r_j_cm**2) * delta_z_cm

def normalization_integrand(E_prime, s, E_max_prime):
    if E_prime <= 0: return 0.0
    return (E_prime**(1.0 - s)) * np.exp(-E_prime / E_max_prime)

def calculate_Ke(L_e, delta_V, s, E_min_prime, E_max_prime):
    integral_E, err = spi.quad(normalization_integrand, E_min_prime, E_max_prime, args=(s, E_max_prime))
    Ke = L_e / (delta_V * integral_E)
    return Ke

def Q_injection(E_prime, K_e, s, E_max_prime, E_min_prime):
    if E_prime < E_min_prime: return 0.0
    return K_e * (E_prime**(-s)) * np.exp(-E_prime / E_max_prime)

# IC e Fótons
def calculate_Gamma_e(E_ph, E_prime):
    return 4 * (E_ph * E_prime) / (m_e**2 * c**4)

def solve_internal_integral(Gamma_e, E_ph, E_prime):
    if E_prime <= E_ph: return 0.0
    q_min = E_ph / (Gamma_e * (E_prime - E_ph))
    q_max = 1.0
    if q_min >= q_max or q_min <= 0: return 0.0

    q_grid = np.logspace(np.log10(q_min), np.log10(q_max), 150)

    def integrand_q(q):
        if q <= 0 or q >= 1: return 0.0
        term1 = 2 * q * np.log(q)
        term2 = (1 + 2 * q) * (1 - q)
        term3 = 0.5 * (1 - q) * (q * Gamma_e)**2 / (1 + q * Gamma_e)
        f_q = term1 + term2 + term3
        E_gamma = (q * Gamma_e * E_prime) / (1.0 + q * Gamma_e)
        dE_gamma_dq = (Gamma_e * E_prime) / (1.0 + q * Gamma_e)**2
        return f_q * (E_gamma - E_ph) * dE_gamma_dq

    vals = np.array([integrand_q(q) for q in q_grid])
    return np.trapz(vals * q_grid, x=np.log(q_grid))

def F_q_IC(q, Gamma_e):
    if q <= 0 or q >= 1: return 0.0
    term1 = 2 * q * np.log(q)
    term2 = (1 + 2 * q) * (1 - q)
    term3 = 0.5 * (1 - q) * (q * Gamma_e)**2 / (1 + q * Gamma_e)
    return term1 + term2 + term3

def Q_IC_prime(E_gamma_prime, Ne_interp_func, interp_syn_log, E_max_ele, m_e, c):
    E_ph_min = 1e-5 * eV_to_erg 
    E_ph_max = 1e4 * eV_to_erg  
    
    if E_gamma_prime < E_ph_min: return 0.0 
        
    E_ph_grid = np.logspace(np.log10(E_ph_min), np.log10(E_ph_max), 60)
    
    term_sqrt_min = np.sqrt(E_gamma_prime / E_ph_max + (E_gamma_prime**2) / (m_e**2 * c**4))
    E_ele_min_absolute = (E_gamma_prime / 2.0) + (m_e * c**2 / 2.0) * term_sqrt_min
    E_ele_min_absolute = max(E_ele_min_absolute, E_gamma_prime)
    E_ele_max = E_max_ele * 10.0
    
    if E_ele_min_absolute >= E_ele_max: return 0.0
        
    E_ele_grid = np.logspace(np.log10(E_ele_min_absolute), np.log10(E_ele_max), 60)
    
    E_ph_2d, E_ele_2d = np.meshgrid(E_ph_grid, E_ele_grid, indexing='ij')
    
    log_n_ph = interp_syn_log(np.log10(E_ph_grid))
    n_ph_1d = np.where(log_n_ph > -100, 10**log_n_ph, 0.0)
    n_ph_2d = n_ph_1d[:, np.newaxis] 
    
    log_Ne = Ne_interp_func(np.log10(E_ele_grid))
    Ne_1d = np.where(log_Ne > -90, 10**log_Ne, 0.0)
    Ne_2d = Ne_1d[np.newaxis, :] 
    
    Gamma_e_2d = 4 * E_ph_2d * E_ele_2d / (m_e**2 * c**4)
    gamma_e_2d = E_ele_2d / (m_e * c**2)
    
    denominator = Gamma_e_2d * (E_ele_2d - E_gamma_prime)
    q_2d = np.where(denominator > 0, E_gamma_prime / denominator, 0.0)
    
    term_sqrt_2d = np.sqrt(E_gamma_prime / E_ph_2d + (E_gamma_prime**2) / (m_e**2 * c**4))
    E_ele_min_2d = np.maximum((E_gamma_prime / 2.0) + (m_e * c**2 / 2.0) * term_sqrt_2d, E_gamma_prime)
    
    valid_mask = (q_2d > 0) & (q_2d < 1) & (E_ele_2d >= E_ele_min_2d) & (n_ph_2d > 0) & (Ne_2d > 0)
    
    q_safe = np.where(valid_mask, q_2d, 0.5)
    F_val_2d = np.zeros_like(q_2d)
    
    term1 = 2 * q_safe * np.log(q_safe)
    term2 = (1 + 2 * q_safe) * (1 - q_safe)
    term3 = 0.5 * (1 - q_safe) * (q_safe * Gamma_e_2d)**2 / (1 + q_safe * Gamma_e_2d)
    F_val_2d[valid_mask] = term1[valid_mask] + term2[valid_mask] + term3[valid_mask]
    
    integrand_2d = (n_ph_2d / E_ph_2d) * (Ne_2d / gamma_e_2d**2) * F_val_2d
    int_ele = np.trapz(integrand_2d, x=E_ele_grid, axis=1)
    res = np.trapz(int_ele, x=E_ph_grid)
    
    return (r_e**2 * c) / 2.0 * res

def bessel_integrand(zeta):
    if zeta > 50: return 0.0
    return kv(5.0/3.0, zeta)

def calculate_Psyn(E_ph_prime, E_ele_prime, B_prime, m_e, c, h, e):
    gamma = E_ele_prime / (m_e * c**2)
    E_cr = (3.0 * e * h * B_prime * gamma**2) / (4.0 * np.pi * m_e * c)
    
    zeta_min = E_ph_prime / E_cr
    if zeta_min > 50.0: return 0.0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", spi.IntegrationWarning)
        val_int, err = spi.quad(bessel_integrand, zeta_min, 100.0)
        
    pre_factor = (np.sqrt(3.0) * e**3 * B_prime) / (m_e * c**2 * h)
    return pre_factor * (E_ph_prime / E_cr) * val_int

def calculate_derivative_term(E_prime_erg, Ne_interp_func):
    log_E = np.log10(E_prime_erg)
    h_step = 1e-4 
    E1, E2 = 10**(log_E - h_step), 10**(log_E + h_step)
    try:
        log_Ne1, log_Ne2 = Ne_interp_func(np.log10(E1)), Ne_interp_func(np.log10(E2))
        if log_Ne1 < -90 or log_Ne2 < -90: return 0.0
        f1, f2 = (10**log_Ne1) / (E1**2), (10**log_Ne2) / (E2**2)
        return (f2 - f1) / (E2 - E1)
    except ValueError: return 0.0

def emissivity_integrand(E_ele_prime, E_ph_prime, K_e, s, E_max_ele, m_e, B_prime, c, sigma_T, t_ad_val, n_c_prime, h, e, Ne_interp_func):
    log_E = np.log10(E_ele_prime)
    log_Ne = Ne_interp_func(log_E)
    if log_Ne <= -90: return 0.0
    Psyn_val = calculate_Psyn(E_ph_prime, E_ele_prime, B_prime, m_e, c, h, e)
    return Psyn_val * (10**log_Ne)

def alpha_SSA_integrand(E_ele, E_ph_prime, B_prime, m_e, c, h, e, Ne_interp_func):
    Psyn_val = calculate_Psyn(E_ph_prime, E_ele, B_prime, m_e, c, h, e)
    if Psyn_val <= 0: return 0.0
    derivative_val = calculate_derivative_term(E_ele, Ne_interp_func)
    if derivative_val == 0.0: return 0.0
    return (E_ele**2) * Psyn_val * derivative_val

def calculate_alpha_SSA(E_ph_prime, B_prime, m_e, c, h, e, Ne_interp_func, lower_lim, E_max_ele_erg):
    pre_factor = -(h**3 * c**2) / (8 * np.pi * E_ph_prime**2)
    upper_lim = E_max_ele_erg * 10.0 
    E_ele_grid = np.logspace(np.log10(lower_lim), np.log10(upper_lim), 150)
    vals = np.array([alpha_SSA_integrand(E_ele, E_ph_prime, B_prime, m_e, c, h, e, Ne_interp_func) for E_ele in E_ele_grid])
    return pre_factor * np.trapz(vals * E_ele_grid, x=np.log(E_ele_grid))

def calculate_n_ph_prime(E_ph_prime, r_j_cm, K_e, s, E_max_ele, m_e, B_prime, c, sigma_T, t_ad_val, n_c_prime, h, e, E_min_ele_erg, Ne_interp_func):
    lower_lim = m_e * c**2
    upper_lim = E_max_ele * 10.0 
    
    E_ele_grid = np.logspace(np.log10(lower_lim), np.log10(upper_lim), 150)
    vals = np.array([emissivity_integrand(E_ele, E_ph_prime, K_e, s, E_max_ele, m_e, B_prime, c, sigma_T, t_ad_val, n_c_prime, h, e, Ne_interp_func) for E_ele in E_ele_grid])
    
    eps_syn = 4 * np.pi * np.trapz(vals * E_ele_grid, x=np.log(E_ele_grid))
    if eps_syn <= 0: return 0.0

    tau_SSA = calculate_alpha_SSA(E_ph_prime, B_prime, m_e, c, h, e, Ne_interp_func, lower_lim, E_max_ele) * r_j_cm 
    
    if tau_SSA < 1e-5: correction_factor = 1.0
    elif tau_SSA > 100: correction_factor = 1.0 / tau_SSA
    else: correction_factor = (1.0 - np.exp(-tau_SSA)) / tau_SSA
        
    return (eps_syn * correction_factor / E_ph_prime) * (r_j_cm / c)

def get_n_ph_val(E, interp_obj):
    if E <= 0: return 0.0
    log_E = np.log10(E)
    log_n = interp_obj(log_E)
    return 10**log_n if log_n > -100 else 0.0

def generate_photon_table(r_j_val, K_e_val, s, E_max_erg, m_e, B_prime_val, c, sigma_T, t_ad_val, n_c_prime, h, e, E_min_erg, Ne_interp_func, PH_Ei, PH_Ef):
    print("   -> [Fótons] Generating photon table (Sync + IC)...")
    E_ph_grid_erg = np.logspace(PH_Ei, PH_Ef, 1000) * eV_to_erg
    
    print("      Calculando componente Síncrotron...")
    n_arr_syn = np.array([calculate_n_ph_prime(E_ph, r_j_val, K_e_val, s, E_max_erg, m_e, B_prime_val, c, sigma_T, t_ad_val, n_c_prime, h, e, E_min_erg, Ne_interp_func) for E_ph in E_ph_grid_erg])
    n_arr_syn[n_arr_syn < 1e-100] = 1e-100
    
    interp_syn_log = interp1d(np.log10(E_ph_grid_erg), np.log10(n_arr_syn), kind='linear', fill_value=-100.0, bounds_error=False)
    
    print("      Calculando componente Compton Inverso (IC)...")
    n_ph_total_list = []
    for i, E_ph in enumerate(E_ph_grid_erg):
        val_syn = n_arr_syn[i]
        val_ic = 0.0
        if E_ph > 1e-3 * eV_to_erg: 
            val_ic = (4.0 * np.pi * Q_IC_prime(E_ph, Ne_interp_func, interp_syn_log, E_max_erg, m_e, c)) * (r_j_val / c)
        n_ph_total_list.append(val_syn + val_ic)
        if (i+1) % 40 == 0: print(f"        Progress: {i+1}/1000")
        
    n_arr_total = np.array(n_ph_total_list)
    n_arr_total[n_arr_total < 1e-100] = 1e-100
    
    interp_total_log = interp1d(np.log10(E_ph_grid_erg), np.log10(n_arr_total), kind='linear', fill_value=-100.0, bounds_error=False)
    return interp_total_log

def IC_rate_numerical_approx0(E_ele_prime, n_ph_function, m_e, c, sigma_T, E_ph_min_integ):
    const_term = (3.0 * m_e**2 * c**5 * sigma_T) / (4.0 * E_ele_prime**3)
    if E_ph_min_integ >= E_ele_prime: return 0.0
    
    E_ph_grid = np.logspace(np.log10(E_ph_min_integ), np.log10(E_ele_prime), 150)
    vals = np.array([(get_n_ph_val(E_ph, n_ph_function) / E_ph) * solve_internal_integral(calculate_Gamma_e(E_ph, E_ele_prime), E_ph, E_ele_prime) if get_n_ph_val(E_ph, n_ph_function) > 0 else 0.0 for E_ph in E_ph_grid])
    
    return const_term * np.trapz(vals * E_ph_grid, x=np.log(E_ph_grid))

def generate_IC_rate_table(E_GeV_array, rate_ic_num_array, Gamma_b, beta, m_e, c, thetaR):
    print("   -> [IC] Generating IC rate table...")
    E_arr = np.array([calculate_E_prime(Gamma_b, E_g * gev_to_erg, beta, m_e, c, thetaR) for E_g in E_GeV_array])
    interp_log = interp1d(np.log10(E_arr), np.log10(np.maximum(rate_ic_num_array, 1e-100)), kind='linear', fill_value=-100.0, bounds_error=False)
    
    def ic_rate_interp_func(E_prime_erg):
        if E_prime_erg <= 0: return 0.0
        log_rate = interp_log(np.log10(E_prime_erg))
        return 10**log_rate if log_rate > -90 else 0.0
    return ic_rate_interp_func

# Transporte
def Q_prime_z(E_prime, z, Ki, s, E_max_prime, z_acc, z_end):
    if z < z_acc or z > z_end or E_prime <= 0: return 0.0
    return Ki * (z_acc / z)**2 * E_prime**(-s) * np.exp(-E_prime / E_max_prime)

def b_total_z_electron(E_prime, z, p, ic_rate_func=None):
    Bz   = calculate_B_field(p['B0'], z, p['z0'], p['m_idx'])
    Bpr  = calculate_B_prime(Bz, p['Gamma_b'])
    nc_p = p['n_c_zacc'] * (p['z_acc'] / z)**2
    
    loss = t_sync_rate(m_e, Bpr, E_prime, c, sigma_T) * E_prime
    loss += t_adiabatic_rate(z, p['beta'], c) * E_prime
    
    gam = E_prime / (m_e * c**2)
    if gam > 1.0: loss += 4*(1.0/137)*sigma_T * nc_p * c * (np.log(2*gam) - 1.0/3.0) * E_prime
    if ic_rate_func is not None: loss += ic_rate_func(E_prime) * E_prime
    return loss

def b_total_z_proton(E_prime, z, p, n_ph_func=None):
    Bz   = calculate_B_field(p['B0'], z, p['z0'], p['m_idx'])
    Bpr  = calculate_B_prime(Bz, p['Gamma_b'])
    nc_p = p['n_c_zacc'] * (p['z_acc'] / z)**2
    
    loss = t_sync_rate_pro(m_p, m_e, Bpr, E_prime, c, sigma_T) * E_prime
    loss += t_adiabatic_rate(z, p['beta'], c) * E_prime
    loss += pp_rates(nc_p, c, E_prime) * E_prime
    
    if n_ph_func is not None:
        loss += t_pg_rate_eq24(E_prime, n_ph_func, c, m_p, E_ph_min_erg=p.get('E_ph_min_erg'), E_ph_max_erg=p.get('E_ph_max_erg')) * E_prime
    return loss

def compute_N_transport(E_prime, z_obs, b_func, Ki, s, E_max_prime, z_acc, z_end, v_b, p, extra_func=None, n_E_steps=200):
    if E_prime >= E_max_prime * 15: return 0.0
    E_grid = np.logspace(np.log10(E_prime), np.log10(E_max_prime * 15), n_E_steps)

    def ode(E_c, zc_arr):
        b = b_func(E_c, zc_arr[0], p, extra_func)
        return [-v_b / b] if b > 0 else [0.0]

    sol = solve_ivp(ode, t_span=(E_grid[0], E_grid[-1]), y0=[z_obs], t_eval=E_grid, method='RK45', rtol=1e-3, atol=0.0)

    integrand = np.zeros(len(sol.t))
    for idx, (Ec, zc) in enumerate(zip(sol.t, sol.y[0])):
        if z_acc <= zc <= z_end:
            Q_val = Q_prime_z(Ec, zc, Ki, s, E_max_prime, z_acc, z_end)
            if Q_val > 0.0:
                b_val = b_func(Ec, zc, p, extra_func)
                if b_val > 0.0: integrand[idx] = Q_val / b_val 

    return max(float(np.trapz(integrand, x=sol.t)), 0.0)

def build_N_grid(E_grid_erg, z_grid_cm, species, tp, extra_func=None, verbose=True):
    b_func = b_total_z_electron if species == 'electron' else b_total_z_proton
    Ki     = tp['K_e'] if species == 'electron' else tp['K_p']
    
    N_grid = np.zeros((len(E_grid_erg), len(z_grid_cm)))
    tasks = [(i, j, E, z) for j, z in enumerate(z_grid_cm) for i, E in enumerate(E_grid_erg)]
    
    if verbose: print(f"   [Transport] Executando paralelização para {species}...")

    def compute_single_point(i, j, E, z):
        return (i, j, compute_N_transport(E, z, b_func, Ki, tp['s'], tp['E_max'], tp['z_acc'], tp['z_end'], tp['v_b'], tp, extra_func))
    
    num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4))

    results = Parallel(n_jobs=num_cores, backend="loky", verbose=0)(
        delayed(compute_single_point)(i, j, E, z) for i, j, E, z in tasks
    )
    for i, j, val in results: N_grid[i, j] = val

    log_E, log_z = np.log10(E_grid_erg), np.log10(z_grid_cm)
    log_N  = np.log10(np.where(N_grid > 1e-300, N_grid, 1e-300))
    
    interp_2d = RegularGridInterpolator((log_E, log_z), log_N, method='linear', bounds_error=False, fill_value=-300.0)
        
    def N_interp_2d(E_erg, z_cm):
        E_arr, z_arr = np.atleast_1d(E_erg), np.atleast_1d(z_cm)
        if len(z_arr) == 1 and len(E_arr) > 1: z_arr = np.full_like(E_arr, float(z_arr[0]))
        pts = np.column_stack((np.log10(E_arr), np.log10(z_arr)))
        res = 10.0 ** interp_2d(pts)
        return float(res[0]) if np.isscalar(E_erg) and np.isscalar(z_cm) else res
        
    def Ne_at_zacc(E_log10):
        vals = N_interp_2d(10.0**E_log10, tp['z_end'])
        if np.isscalar(E_log10): return np.log10(vals) if vals > 0 else -300.0
        return np.where(vals > 0, np.log10(vals), -300.0)
        
    return N_grid, N_interp_2d, Ne_at_zacc

# Prótons Física Focada
def t_sync_rate_pro(m_p, m_e, B_prime, E_prime, c, sigma_T):
    return t_sync_rate(m_e, B_prime, E_prime, c, sigma_T) * (m_e/m_p)**4

def pp_rates(n_c_prime, c, E_prime_erg):
    E_p = E_prime_erg / gev_to_erg
    E_th = 1.22 
    E_p_safe = np.maximum(E_p, E_th + 1e-5)
    L = np.log(E_p_safe/1000.0)
    sigmap = (34.3 + 1.88*L + 0.25*L**2) * ((1.0 - (E_th/E_p_safe)**4)**2) * 1e-27
    pp = n_c_prime * c * sigmap * 0.5
    return np.where(E_p <= E_th, 1e-30, pp)

def K_pairs(x_prime):
    if x_prime <= 2.0: return 0.0
    if x_prime < 1000:
        term_ln = np.log(x_prime - 1)
        brac = 1 + 0.3957*term_ln + 0.1*(term_ln**2) + 0.0078*(term_ln**3)
        return 4 * (m_e/m_p) * (1/x_prime) * brac
    ln_x, ln_2x = np.log(x_prime), np.log(2*x_prime)
    num = -8.78 + 5.513*ln_x - 1.612*(ln_x**2) + 0.668*(ln_x**3)
    den = 3.1111*ln_2x - 8.0741
    return 4 * (m_e/m_p) * (1/x_prime) * (num/den)

def sigma_pairs(x_prime):
    if x_prime < 4.0:
        eta = (x_prime - 2)/(x_prime + 2)
        series = 1 + 0.5*eta + (23/40)*(eta**2) + (37/120)*(eta**3) + (61/192)*(eta**4)
        return 1.2135e-27 * ((x_prime - 2)/2)**3 * series
    ln_2x = np.log(2*x_prime)
    term1 = 3.1111*ln_2x - 8.0741
    term2 = (2/x_prime)**2 * (2.7101*ln_2x - (ln_2x**2) + 0.6667*(ln_2x**3) + 0.5490)
    return 5.7938e-28 * (term1 + term2)

def get_sigma_K_pion(x_prime):
    if x_prime < 284: return 0.0
    K_pi = 9.78e-3 * (x_prime**0.4756)
    if 284 <= x_prime < 644: sigma_pi = 5.48e-28 * (x_prime/644)**2.60
    elif 644 <= x_prime < 978: sigma_pi = 5.48e-28 * (x_prime/644)**(-2.89)
    elif 978 <= x_prime < 1370: sigma_pi = 1.64e-28 * (x_prime/978)**1.51
    elif 1370 <= x_prime < 2387: sigma_pi = 2.73e-28 * (x_prime/1370)**(-1.20)
    else: return 1.4e-27 * 0.5 
    return sigma_pi * K_pi

def t_pg_rate_eq24(E_p_erg, n_ph_func, c, m_p, E_ph_min_erg=None, E_ph_max_erg=None):
    gamma_p = E_p_erg / (m_p * c**2)
    lower = max((2.0 * mec2_erg) / (2.0 * gamma_p), E_ph_min_erg) if E_ph_min_erg else (2.0 * mec2_erg) / (2.0 * gamma_p)
    upper = E_ph_max_erg or 1.0e8 * eV_to_erg   
    if lower >= upper: return 1e-30

    def integrando_eq24_interno(epsilon_r):
        x_prime = epsilon_r / mec2_erg
        return (get_sigma_K_pion(x_prime) + (sigma_pairs(x_prime) * K_pairs(x_prime))) * epsilon_r

    def integrando_eq24_externo(E_ph_erg):
        n_density = get_n_ph_val(E_ph_erg, n_ph_func)
        if n_density <= 0.0: return 0.0
        epsilon_max = 2.0 * gamma_p * E_ph_erg
        if epsilon_max <= 2.0 * mec2_erg: return 0.0
        
        pts = [p for p in [284.0 * mec2_erg, 644.0 * mec2_erg, 978.0 * mec2_erg] if 2.0 * mec2_erg < p < epsilon_max]
        val_interna, _ = spi.quad(integrando_eq24_interno, 2.0 * mec2_erg, epsilon_max, points=pts, limit=300, epsabs=0, epsrel=1e-3)
        return (n_density / E_ph_erg**2) * val_interna

    E_ph_grid = np.logspace(np.log10(lower), np.log10(upper), 500)
    vals = np.array([integrando_eq24_externo(E_ph) for E_ph in E_ph_grid])
    rate = (c / (2.0 * gamma_p**2)) * np.trapz(vals * E_ph_grid, x=np.log(E_ph_grid))
    return max(rate, 1e-30)

def t_shear_rate(Gamma_j, beta_j, delta_r, B_prime, E_prime, q, c):
    delta_u = beta_j * c
    r_g = E_prime / (q * B_prime)
    return 1.0 / np.maximum((3.0 * (delta_r**2) * c) / ((Gamma_j**4) * (delta_u**2) * r_g), r_g / c)

def tau_escape(E_erg, q, B_prime, c, delta_r):
    r_g = E_erg / (q * B_prime)
    return (1.5 * delta_r**2) / (r_g*c)

# Setup de Parâmetros Globais
def converter(valor_str):
    valor_str = str(valor_str).lower()
    if 'e' in valor_str:
        base, expoente = valor_str.split('e')
        return float(base) * (10 ** float(expoente))
    return float(valor_str)

def get_base_params(params):
    M = converter(params['M'])
    Gamma_b_val = float(params['Gamma_b'])
    zetaR = np.radians(float(params['xi_j']))
    
    Z_acc_rg_val = calculate_z_acc(Z_0_rg, float(params['q_m']), float(params['m'])) 
    Rg = M * 1.477e5  
    Z_acc = Z_acc_rg_val * Rg          
    delta_z_cm = (2 * Z_acc_rg_val * np.tan(zetaR)) * Rg
    beta_val = np.sqrt(1 - 1/Gamma_b_val**2)
    v_b = beta_val * c

    r_j_val_z = jet_radius(Z_acc, zetaR)
    B_0_val = calculate_B0(float(params['L_j']), jet_radius(Z_0_rg * Rg, zetaR), v_b)
    B_prime_val = calculate_B_prime(calculate_B_field(B_0_val, Z_acc, Z_0_rg * Rg, float(params['m'])), Gamma_b_val)
    n_c_prime = calculate_nc(Z_acc, float(params['L_j']), Gamma_b_val, float(params['xi_j']), float(params['q_rel'])) / Gamma_b_val

    E_min_ele, E_max_ele = float(params['E_min_gev']) * gev_to_erg, float(params['E_max_gev']) * gev_to_erg
    K_e_val = calculate_Ke(calculate_Le(float(params['L_j']), float(params['q_rel']), float(params['a'])), calculate_Delta_V(r_j_val_z, delta_z_cm), float(params['s']), E_min_ele, E_max_ele)

    return {
        'M': M, 'q_m': float(params['q_m']), 'L_j': float(params['L_j']), 'Gamma_b': Gamma_b_val, 
        'theta_rad': np.radians(float(params['theta'])), 'm_idx': float(params['m']), 
        'n': float(params['n']), 'n_2': float(params.get('n_2', 100.0)), 's': float(params['s']), 
        'q_rel': float(params['q_rel']), 'a': float(params['a']), 'E_min_ele_erg': E_min_ele, 
        'E_max_ele_erg': E_max_ele, 'PH_Ei': float(params['PH_Ei']), 'PH_Ef': float(params['PH_Ef']),
        'z0': Z_0_rg * Rg, 'Z_acc_cm': Z_acc, 'delta_z_cm': delta_z_cm, 'beta': beta_val, 'v_b_cm_s': v_b, 
        'r_j_z0_cm': r_j_val_z, 'B_0_Gauss': B_0_val, 'B_prime_Gauss': B_prime_val, 
        't_ad_s': t_adiabatic_rate(Z_acc, beta_val, c), 'n_c_prime': n_c_prime, 
        'Delta_V_vol': calculate_Delta_V(r_j_val_z, delta_z_cm), 'K_e_norm': K_e_val
    }

def build_transport_params(base_p, K_e, K_p, E_max_ele, E_max_pro, E_ph_min_erg=None, E_ph_max_erg=None):
    return {
        'Gamma_b': base_p['Gamma_b'], 'beta': base_p['beta'], 'v_b': base_p['v_b_cm_s'],
        'z0': base_p['z0'], 'z_acc': base_p['Z_acc_cm'], 'z_end': base_p['Z_acc_cm'] + base_p['delta_z_cm'],
        'B0': base_p['B_0_Gauss'], 'm_idx': base_p['m_idx'], 'n_c_zacc': base_p['n_c_prime'],
        's': base_p['s'], 'K_e': K_e, 'K_p': K_p, 'E_max': E_max_ele,
        'E_ph_min_erg': E_ph_min_erg, 'E_ph_max_erg': E_ph_max_erg
    }

# Simulação: Elétrons
def process_and_save_electrons(source_name, base_p, output_dir="data"):
    start_time = time.time()
    print(f"[{source_name}] Calculating Electrons (1D Transport)...")
    
    E_GeV_ele = np.logspace(-4, 14, 1000)
    E_prime_erg_ele = [calculate_E_prime(base_p['Gamma_b'], E_g*gev_to_erg, base_p['beta'], m_e, c, base_p['theta_rad']) for E_g in E_GeV_ele]

    z_grid = np.concatenate((np.linspace(base_p['Z_acc_cm'], base_p['Z_acc_cm'] + base_p['delta_z_cm'], 20),
                             np.logspace(np.log10(base_p['Z_acc_cm'] + base_p['delta_z_cm'] * 1.05), np.log10(100.0 * base_p['Z_acc_cm']), 20)))
    E_grid_ele = np.logspace(np.log10(base_p['E_min_ele_erg'] * 0.5), np.log10(base_p['E_max_ele_erg'] * 20), 40)
    tp = build_transport_params(base_p, base_p['K_e_norm'], 0, base_p['E_max_ele_erg'], base_p['E_max_ele_erg'])

    print(f"   [{source_name}] -> Transport approx 0 (sem IC)...")
    _, _, Ne_at_zacc_0 = build_N_grid(E_grid_ele, z_grid, 'electron', tp, None, True)

    n_ph_func_approx0 = generate_photon_table(
        base_p['r_j_z0_cm'], base_p['K_e_norm'], base_p['s'], base_p['E_max_ele_erg'], m_e, base_p['B_prime_Gauss'], c,
        sigma_T, base_p['t_ad_s'], base_p['n_c_prime'], h, e, base_p['E_min_ele_erg'], Ne_at_zacc_0, base_p['PH_Ei'], base_p['PH_Ef'])

    E_ph_min_integ = 1e-8 * eV_to_erg
    ic_rate_func_approx0 = generate_IC_rate_table(
        E_GeV_ele, [IC_rate_numerical_approx0(Ep, n_ph_func_approx0, m_e, c, sigma_T, E_ph_min_integ) for Ep in E_prime_erg_ele],
        base_p['Gamma_b'], base_p['beta'], m_e, c, base_p['theta_rad'])

    print(f"   [{source_name}] -> Transport approx 1 (com IC)...")
    _, N_interp_2d_1, Ne_at_zacc_1 = build_N_grid(E_grid_ele, z_grid, 'electron', tp, ic_rate_func_approx0, True)

    n_ph_func_approx1 = generate_photon_table(
        base_p['r_j_z0_cm'], base_p['K_e_norm'], base_p['s'], base_p['E_max_ele_erg'], m_e, base_p['B_prime_Gauss'], c,
        sigma_T, base_p['t_ad_s'], base_p['n_c_prime'], h, e, base_p['E_min_ele_erg'], Ne_at_zacc_1, base_p['PH_Ei'], base_p['PH_Ef'])

    E_ph_grid_eV = np.logspace(base_p['PH_Ei'], base_p['PH_Ef'], 1000)
    with open(f"{output_dir}/{source_name}_photons.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['E_ph_eV', 'E_ph_erg', 'n_ph_density'])
        for E_eV in E_ph_grid_eV: writer.writerow([E_eV, E_eV * eV_to_erg, get_n_ph_val(E_eV * eV_to_erg, n_ph_func_approx1)])

    E_prime_arr_ele = np.array(E_prime_erg_ele)
    rate_acc_ele = t_acc_rate(base_p['n'], c, e, base_p['B_prime_Gauss'], E_prime_arr_ele)
    rate_acc_ele_2 = t_acc_rate(base_p['n_2'], c, e, base_p['B_prime_Gauss'], E_prime_arr_ele)
    rate_sync_ele = t_sync_rate(m_e, base_p['B_prime_Gauss'], E_prime_arr_ele, c, sigma_T)
    rate_bremss = t_bremss_rate(E_prime_arr_ele, base_p['n_c_prime'], c)
    rate_shear_ele = t_shear_rate(base_p['Gamma_b'], base_p['beta'], base_p['r_j_z0_cm'], base_p['B_prime_Gauss'], E_prime_arr_ele, e, c)
    rate_esc_ele = tau_escape(E_prime_arr_ele, e, base_p['B_prime_Gauss'], c, base_p['r_j_z0_cm'])
    rate_ic_num_approx1 = np.array([IC_rate_numerical_approx0(Ep, n_ph_func_approx1, m_e, c, sigma_T, E_ph_min_integ) for Ep in E_prime_arr_ele])

    with open(f"{output_dir}/{source_name}_electrons.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['E_GeV', 'rate_acc', 'rate_acc_2', 'rate_sync', 'rate_bremss', 'rate_ic_0', 'rate_ic_1', 't_ad_val', 'Shear', 'ESC'])
        for i in range(len(E_GeV_ele)):
            writer.writerow([E_GeV_ele[i], rate_acc_ele[i], rate_acc_ele_2[i], rate_sync_ele[i], rate_bremss[i], ic_rate_func_approx0(E_prime_arr_ele[i]), rate_ic_num_approx1[i], base_p['t_ad_s'], rate_shear_ele[i], rate_esc_ele[i]])
    
    mins, secs = divmod(time.time() - start_time, 60)
    print(f"   === [Elétrons] Concluído em {int(mins)}m {secs:.2f}s ===")
    return n_ph_func_approx1

# Simulação: Prótons
def process_and_save_protons(source_name, base_p, n_ph_func_approx1, output_dir="data"):
    start_time = time.time()
    print(f"[{source_name}] Calculating Protons ")
    
    z_grid = np.logspace(np.log10(base_p['Z_acc_cm'] * 1.001), np.log10(100.0 * base_p['Z_acc_cm']), 20)
    E_max_pro = base_p.get('E_max_pro_erg', 2e9 * gev_to_erg)
    K_p_val = calculate_Ke(base_p['Delta_V_vol'] * base_p['a'] * base_p['K_e_norm'], base_p['Delta_V_vol'], base_p['s'], 3*gev_to_erg, E_max_pro) # Simplificado o cálculo de Lp para usar Ke
    
    E_grid_pro = np.logspace(np.log10(1.22 * gev_to_erg), np.log10(E_max_pro * 20), 20)
    E_ph_min, E_ph_max = (10.0 ** base_p['PH_Ei']) * eV_to_erg, (10.0 ** base_p['PH_Ef']) * eV_to_erg
    tp_pro = build_transport_params(base_p, 0, K_p_val, E_max_pro, E_max_pro, E_ph_min, E_ph_max)

    print(f"   [{source_name}] -> Pré-calculando tabelas p-gamma...")
    E_p_grid_opt = np.logspace(np.log10(1.22 * gev_to_erg), np.log10(E_max_pro * 50), 150)
    
    # 1. Calculamos o t_pg E O OMEGA ao mesmo tempo
    t_pg_base_vals = []
    omega_base_vals = []
    for Ep in tqdm(E_p_grid_opt, desc="Gerando Tabelas", unit="step"):
        t_pg_base_vals.append(t_pg_rate_eq24(Ep, n_ph_func_approx1, c, m_p, E_ph_min, E_ph_max))
        omega_base_vals.append(omega_pg_pion(Ep, n_ph_func_approx1, c, m_p, E_ph_max))
    
    # 2. Salvamos as duas matrizes DENTRO da base_p para não perder
    base_p['interp_tpg_base'] = interp1d(np.log10(E_p_grid_opt), np.log10(np.maximum(t_pg_base_vals, 1e-60)), bounds_error=False, fill_value=-60.0)
    base_p['interp_omega_base'] = interp1d(np.log10(E_p_grid_opt), np.log10(np.maximum(omega_base_vals, 1e-60)), bounds_error=False, fill_value=-60.0)
    
    def t_pg_interp_func(E_prime):
        return 10**base_p['interp_tpg_base'](np.log10(E_prime)) if E_prime > 0 and base_p['interp_tpg_base'](np.log10(E_prime)) > -55 else 0.0

    print(f"   [{source_name}] -> Transport protons...")
    N_grid_pro, N_pro_interp_2d, _ = build_N_grid(E_grid_pro, z_grid, 'proton', tp_pro, t_pg_interp_func, True)
    
    base_p['N_pro_interp_2d'] = N_pro_interp_2d
    
    # 2. Salvamos a matriz bruta diretamente
    np.save(f"{output_dir}/{source_name}_protons_N_grid.npy", N_grid_pro)
    np.save(f"{output_dir}/{source_name}_protons_E_grid.npy", E_grid_pro)
    np.save(f"{output_dir}/{source_name}_protons_z_grid.npy", z_grid)
    E_GeV_pro = np.logspace(0, 20, 500)
    E_prime_arr_pro = np.array([calculate_E_prime(base_p['Gamma_b'], E_g*gev_to_erg, base_p['beta'], m_p, c, base_p['theta_rad']) for E_g in E_GeV_pro])
    
    rate_acc_pro = t_acc_rate(base_p['n'], c, e, base_p['B_prime_Gauss'], E_prime_arr_pro)
    rate_acc_pro_2 = t_acc_rate(base_p['n_2'], c, e, base_p['B_prime_Gauss'], E_prime_arr_pro)
    rate_sync_pro = t_sync_rate_pro(m_p, m_e, base_p['B_prime_Gauss'], E_prime_arr_pro, c, sigma_T)
    rate_pp_pro = pp_rates(base_p['n_c_prime'], c, E_prime_arr_pro)
    rate_shear_pro = t_shear_rate(base_p['Gamma_b'], base_p['beta'], base_p['r_j_z0_cm'], base_p['B_prime_Gauss'], E_prime_arr_pro, e, c)
    rate_esc_pro = tau_escape(E_prime_arr_pro, e, base_p['B_prime_Gauss'], c, base_p['r_j_z0_cm'])
    rate_pg_pro = np.array([t_pg_interp_func(Ep) for Ep in E_prime_arr_pro])

    with open(f"{output_dir}/{source_name}_protons.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['E_prime_GeV', 'rate_acc', 'rate_acc_2', 'rate_sync', 'rate_pp', 'rate_pg', 't_ad_val', 'Shear', 'ESC'])
        for i in range(len(E_GeV_pro)):
            writer.writerow([E_prime_arr_pro[i] / gev_to_erg, rate_acc_pro[i], rate_acc_pro_2[i], rate_sync_pro[i], rate_pp_pro[i], rate_pg_pro[i], base_p['t_ad_s'], rate_shear_pro[i], rate_esc_pro[i]])

    mins, secs = divmod(time.time() - start_time, 60)
    print(f"   === [Prótons] Concluído em {int(mins)}m {secs:.2f}s ===")

def sigma_pion_only(x_prime):
    if x_prime < 284: return 0.0
    if 284 <= x_prime < 644:
        return 5.48e-28 * (x_prime/644)**2.60
    elif 644 <= x_prime < 978:
        return 5.48e-28 * (x_prime/644)**(-2.89)
    elif 978 <= x_prime < 1370:
        return 1.64e-28 * (x_prime/978)**1.51
    elif 1370 <= x_prime < 2387:
        return 2.73e-28 * (x_prime/1370)**(-1.20)
    else:
        return 1.4e-27

def omega_pg_pion(E_p_erg, n_ph_func, c, m_p, E_ph_max_erg):
    gamma_p = E_p_erg / (m_p * c**2)
    epsilon_th_pion = 284.0 * mec2_erg
    E_ph_threshold = epsilon_th_pion / (2.0 * gamma_p)
    lower = max(E_ph_threshold, 1e-5 * eV_to_erg) 
    
    if lower >= E_ph_max_erg: return 1e-30

    def integrando_interno_omega(E_ph_erg):
        epsilon_max = 2.0 * gamma_p * E_ph_erg
        if epsilon_max <= epsilon_th_pion: return 0.0

        def integrand_eps(eps):
            x_prime = eps / mec2_erg
            return sigma_pion_only(x_prime) * eps

        pts = [644.0*mec2_erg, 978.0*mec2_erg]
        valid_pts = [p for p in pts if epsilon_th_pion < p < epsilon_max]
        
        # Silenciador de erros de integração para o degrau do Píon
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", spi.IntegrationWarning)
            val, _ = spi.quad(integrand_eps, epsilon_th_pion, epsilon_max, points=valid_pts, limit=100)
        return val

    def integrando_externo_omega(E_ph):
        n_density = get_n_ph_val(E_ph, n_ph_func)
        if n_density <= 0: return 0.0
        val_int = integrando_interno_omega(E_ph)
        return (n_density / E_ph**2) * val_int

    E_ph_grid = np.logspace(np.log10(lower), np.log10(E_ph_max_erg), 150)
    vals = np.array([integrando_externo_omega(E) for E in E_ph_grid])

    # Integração logarítmica
    integral_log = np.trapz(vals * E_ph_grid, x=np.log(E_ph_grid))
    rate = (c / (2.0 * gamma_p**2)) * integral_log
    
    return rate if rate > 1e-30 else 1e-30

# =========================================================================
# FÍSICA DE PÍONS E MÚONS (DECAIMENTO E TRANSPORTE ALGÉBRICO)
# =========================================================================
m_pi_GeV = 0.1396
m_pi_erg = m_pi_GeV * gev_to_erg
tau_0_pi = 2.6e-8 

m_mu_GeV = 0.10566
m_mu_erg = m_mu_GeV * gev_to_erg
r_pi = (m_mu_GeV / 0.1396)**2  
tau_0_mu = 2.2e-6 

def T_decay_pion(E_pi_prime_erg):
    return (E_pi_prime_erg / m_pi_erg) * tau_0_pi

def T_decay_muon(E_mu_prime_erg):
    return (E_mu_prime_erg / m_mu_erg) * tau_0_mu

def sigma_pp_inel(E_p_GeV):
    E_th = 1.22 
    if E_p_GeV <= E_th: return 0.0
    L = np.log(E_p_GeV / 1000.0)
    return (34.3 + 1.88 * L + 0.25 * L**2) * ((1 - (E_th / E_p_GeV)**4)**2) * 1e-27

def F_pi_kelner(x, E_p_GeV):
    if x <= 0 or x >= 1 or E_p_GeV <= 1.22: return 0.0
    L = np.log(E_p_GeV / 1000.0)
    a_prime = 3.67 + 0.83 * L + 0.075 * L**2
    B_pi, r_prime, alpha = a_prime + 0.25, 2.6 / np.sqrt(a_prime), 0.98 / np.sqrt(a_prime)
    E_pi_GeV = x * E_p_GeV
    if E_pi_GeV <= m_pi_GeV: return 0.0
    
    term1 = 4 * alpha * B_pi * (x**(alpha - 1.0))
    term2 = ((1.0 - x**alpha) / (1.0 + r_prime * x**alpha * (1.0 - x**alpha)))**4
    term3 = (1.0 / (1.0 - x**alpha)) + (r_prime * (1.0 - 2 * x**alpha)) / (1.0 + r_prime * x**alpha * (1.0 - x**alpha))
    term4 = np.sqrt(1.0 - m_pi_GeV / E_pi_GeV)
    return term1 * term2 * term3 * term4

def Q_pi_pp_prime(E_pi_erg, z_cm, nc_prime, N_pro_interp_2d, E_max_pro_erg):
    E_pi_GeV = E_pi_erg / gev_to_erg
    if E_pi_GeV <= m_pi_GeV: return 0.0

    val_low = 0.0
    K_pi, n_pi = 0.17, 2.0  
    E_p_star_GeV = (E_pi_GeV / K_pi) + 0.938
    E_p_star_erg = E_p_star_GeV * gev_to_erg

    if E_p_star_GeV < 100.0 and E_p_star_erg <= E_max_pro_erg:
        N_p_val_low = N_pro_interp_2d(E_p_star_erg, z_cm)
        if N_p_val_low > 0:
            val_low = (n_pi / K_pi) * N_p_val_low * sigma_pp_inel(E_p_star_GeV)

    val_high = 0.0
    x_min = E_pi_erg / E_max_pro_erg
    x_max = min(1.0, E_pi_GeV / 100.0) 

    if x_min < x_max:
        def integrand_high(x):
            if x <= 0: return 0.0
            E_p_GeV = E_pi_GeV / x
            N_p_val = N_pro_interp_2d(E_p_GeV * gev_to_erg, z_cm)
            if N_p_val <= 0: return 0.0
            return (1.0 / x) * N_p_val * F_pi_kelner(x, E_p_GeV) * sigma_pp_inel(E_p_GeV)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", spi.IntegrationWarning)
            val_high, _ = spi.quad(integrand_high, x_min, x_max, limit=200, epsrel=1e-3, epsabs=1e-10)

    return nc_prime * c * (val_low + val_high)


def Q_pi_pg_prime(E_pi_erg, z_cm, N_pro_interp_2d, p_dict):
    E_p_erg = 5.0 * E_pi_erg
    N_p_val = N_pro_interp_2d(E_p_erg, z_cm)
    if N_p_val <= 0: return 0.0

    z_acc = p_dict['Z_acc_cm']
    z_end = z_acc + p_dict['delta_z_cm']
    
    if z_cm > z_end:
        return 0.0
        
    fator_densidade = (z_acc / z_cm)**2
    log_Ep = np.log10(E_p_erg)
    
    # Puxa o OMEGA E O T_PG que nós salvamos no p_dict no passo anterior
    omega_val = (10**p_dict['interp_omega_base'](log_Ep)) * fator_densidade
    if omega_val <= 1e-25: return 0.0

    t_pg_inv = (10**p_dict['interp_tpg_base'](log_Ep)) * fator_densidade

    K_bar = t_pg_inv / omega_val if omega_val > 0 else 0.5
    K_bar = max(0.2, min(0.6, K_bar)) 

    p1 = (0.6 - K_bar) / (0.6 - 0.2)
    p2 = 1.0 - p1
    N_pi_mult = (p1 / 2.0) + 2.0 * p2
    f_corr = 0.007
    
    return 5.0 * N_p_val * omega_val * N_pi_mult * f_corr

def decay_spectra_mu(E_mu_erg, E_pi_erg, helicity='L'):
    x = E_mu_erg / E_pi_erg
    if x < r_pi or x > 1.0: return 0.0
    pre_factor = 1.0 / (E_pi_erg * x * (1.0 - r_pi)**2)
    return pre_factor * r_pi * (1.0 - x) if helicity == 'L' else pre_factor * (x - r_pi)

def Q_mu_prime(E_mu_erg, z_cm, N_pi_interp_2d, E_max_pro_erg, helicity='L'):
    if E_mu_erg <= m_mu_erg: return 0.0
    def integrand(E_pi_erg):
        N_pi_val = N_pi_interp_2d(E_pi_erg, z_cm)
        if N_pi_val <= 0: return 0.0
        return (1.0 / T_decay_pion(E_pi_erg)) * N_pi_val * decay_spectra_mu(E_mu_erg, E_pi_erg, helicity)

    lower_limit = E_mu_erg
    upper_limit = min(E_mu_erg / r_pi, E_max_pro_erg) 
    if lower_limit >= upper_limit: return 0.0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", spi.IntegrationWarning)
        val, _ = spi.quad(integrand, lower_limit, upper_limit, limit=100, epsabs=1e-50, epsrel=1e-3)
    return val

def b_total_z_pion(E_prime, z, p):
    Bz   = calculate_B_field(p['B0'], z, p['z0'], p['m_idx'])
    Bpr  = calculate_B_prime(Bz, p['Gamma_b'])
    loss_sync = t_sync_rate(m_e, Bpr, E_prime, c, sigma_T) * (m_e / (m_pi_erg/c**2))**4 * E_prime
    loss_ad = t_adiabatic_rate(z, p['beta'], c) * E_prime
    return loss_sync + loss_ad

def b_total_z_muon(E_prime, z, p):
    Bz  = calculate_B_field(p['B0'], z, p['z0'], p['m_idx'])
    Bpr = calculate_B_prime(Bz, p['Gamma_b'])
    loss_sync = t_sync_rate(m_e, Bpr, E_prime, c, sigma_T) * (m_e / (m_mu_erg/c**2))**4 * E_prime
    loss_ad   = t_adiabatic_rate(z, p['beta'], c) * E_prime
    return loss_sync + loss_ad

def build_N_decay_grid(E_grid_erg, z_grid_cm, b_func, Q_func, T_decay_func, p_dict, desc="Transport"):
    N_grid = np.zeros((len(E_grid_erg), len(z_grid_cm)))
    E_max_limit = p_dict['E_max']
    
    for j, z in enumerate(z_grid_cm):
        for i, E in enumerate(E_grid_erg):
            if E > E_max_limit * 1.5:
                N_grid[i, j] = 1e-300
                continue
                
            Q_val = Q_func(E, z)
            if Q_val > 1e-300:
                T_d = T_decay_func(E)
                b_val = b_func(E, z, p_dict) 
                rate_decay = 1.0 / T_d
                rate_cool = b_val / E if b_val > 0.0 else 0.0
                N_val = Q_val / (rate_decay + rate_cool)
                
                if E > 0.1 * E_max_limit:
                    N_val *= np.exp(-(E / E_max_limit))
                N_grid[i, j] = max(N_val, 1e-300)
            else:
                N_grid[i, j] = 1e-300

    log_E, log_z = np.log10(E_grid_erg), np.log10(z_grid_cm)
    log_N = np.log10(N_grid)
    interp_2d = RegularGridInterpolator((log_E, log_z), log_N, method='linear', bounds_error=False, fill_value=-300.0)
    
    def N_interp_2d(E_erg, z_cm):
        if E_erg > E_max_limit * 1.5: return 0.0
        pt = np.array([[np.log10(float(E_erg)), np.log10(float(z_cm))]])
        val = float(10.0 ** interp_2d(pt)[0])
        if E_erg > 0.1 * E_max_limit:
            val *= np.exp(-(E_erg / E_max_limit)**2)
        return val
    return N_interp_2d

# =========================================================================
# KELNER & AHARONIAN 2008 - EXATO (p-gamma)
# =========================================================================
_eta_grid_kelner = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 30.0, 100.0])
_B_numu_bar = np.array([0.809, 7.70, 19.9, 36.2, 53.9, 73.9, 94.8, 120, 147, 174, 338, 517, 761, 957, 1110, 1250, 1360, 1460, 5870, 31000])
_s_numu_bar = np.array([0.365, 0.287, 0.250, 0.238, 0.220, 0.206, 0.197, 0.193, 0.187, 0.178, 0.123, 0.106, 0.0944, 0.0829, 0.0801, 0.0752, 0.0680, 0.0615, 0.0361, 0.0228])
_del_numu_bar = np.array([3.09, 2.96, 2.89, 2.76, 2.71, 2.67, 2.62, 2.56, 2.52, 2.51, 2.48, 2.56, 2.57, 2.58, 2.54, 2.53, 2.56, 2.60, 2.78, 2.88])
_B_numu = np.array([1.08, 9.91, 24.7, 44.3, 67.0, 90.4, 118, 132, 177, 211, 383, 509, 726, 926, 1070, 1190, 1290, 1400, 5650, 30100])
_s_numu = np.array([0.0, 0.0778, 0.242, 0.377, 0.440, 0.450, 0.461, 0.451, 0.464, 0.446, 0.366, 0.249, 0.204, 0.174, 0.156, 0.140, 0.121, 0.107, 0.0705, 0.0463])
_del_numu = np.array([0.0, 0.306, 0.792, 1.09, 1.06, 0.953, 0.956, 0.922, 0.912, 0.940, 1.49, 2.03, 2.18, 2.24, 2.28, 2.32, 2.39, 2.46, 2.53, 2.62])

_interp_log_B_bar = interp1d(np.log10(_eta_grid_kelner), np.log10(_B_numu_bar), fill_value='extrapolate')
_interp_s_bar     = interp1d(np.log10(_eta_grid_kelner), _s_numu_bar, fill_value='extrapolate')
_interp_del_bar   = interp1d(np.log10(_eta_grid_kelner), _del_numu_bar, fill_value='extrapolate')
_interp_log_B_nu  = interp1d(np.log10(_eta_grid_kelner), np.log10(_B_numu), fill_value='extrapolate')
_interp_s_nu      = interp1d(np.log10(_eta_grid_kelner), _s_numu, fill_value='extrapolate')
_interp_del_nu    = interp1d(np.log10(_eta_grid_kelner), _del_numu, fill_value='extrapolate')

def get_kelner_params(rho, lepton_type):
    if rho < 1.1: return 0.0, 0.0, 0.0
    rho_safe = min(rho, 100.0)
    log_rho = np.log10(rho_safe)
    fator_conversao = 1e-30 * 2.9979e10 
    
    if lepton_type == 'numu_bar':
        return (10**_interp_log_B_bar(log_rho)) * fator_conversao, _interp_s_bar(log_rho), _interp_del_bar(log_rho)
    else:
        return (10**_interp_log_B_nu(log_rho)) * fator_conversao, _interp_s_nu(log_rho), _interp_del_nu(log_rho)

def Phi_nu_Kelner(eta, x, lepton_type):
    eta0, r = 0.313, 0.146 
    if eta < eta0: return 0.0
    rho = eta / eta0
    
    term_sqrt = np.sqrt(max(0, (eta - r**2 - 2*r) * (eta - r**2 + 2*r)))
    x_minus = (eta + r**2 - term_sqrt) / (2 * (1.0 + eta))
    x_plus  = (eta + r**2 + term_sqrt) / (2 * (1.0 + eta))
    psi = 2.5 + 1.4 * np.log(rho)
    
    if lepton_type == 'numu_bar':
        x_min_prime, x_max_prime = x_minus / 4.0, x_plus
    else:
        x_min_prime = 0.427 * x_minus
        if rho < 2.14: x_max_prime = 0.427 * x_plus
        elif rho < 10.0: x_max_prime = (0.427 + 0.0729 * (rho - 2.14)) * x_plus
        else: x_max_prime = x_plus
            
    B, s, delta = get_kelner_params(rho, lepton_type)
    
    if x < x_min_prime: return B * (np.log(2.0))**psi
    elif x_min_prime <= x < x_max_prime:
        y_prime = (x - x_min_prime) / (x_max_prime - x_min_prime)
        return B * np.exp(-s * (np.log(x / x_min_prime))**delta) * (np.log(2.0 / (1.0 + y_prime**2)))**psi
    else: return 0.0

def calculate_Q_nu_pg_exact(E_nu_erg, z_cm, N_pro_interp_2d, n_ph_interp_obj, base_p):
    E_max_pro = base_p['E_max_pro_erg']
    x_min = E_nu_erg / E_max_pro
    if x_min >= 1.0: return 0.0
    
    x_grid = np.logspace(np.log10(x_min), 0, 70)
    integral_x = []
    fator_diluicao_foton = (base_p['Z_acc_cm'] / z_cm)**2
    
    for x in x_grid:
        E_p_erg = E_nu_erg / x
        Np_val = N_pro_interp_2d(E_p_erg, z_cm)
        if Np_val <= 0:
            integral_x.append(0.0)
            continue
            
        eps_min = 0.313 * ((m_p * c**2)**2) / (4 * E_p_erg)
        eps_max = 1e5 * eV_to_erg 
        if eps_min >= eps_max:
            integral_x.append(0.0)
            continue
            
        eps_grid = np.logspace(np.log10(eps_min), np.log10(eps_max), 70)
        integral_eps = []
        
        for eps in eps_grid:
            n_ph = get_n_ph_val(eps, n_ph_interp_obj) * fator_diluicao_foton
            if n_ph <= 0:
                integral_eps.append(0.0)
                continue
            eta = 4 * eps * E_p_erg / ((m_p * c**2)**2)
            integral_eps.append(n_ph * (Phi_nu_Kelner(eta, x, 'numu') + Phi_nu_Kelner(eta, x, 'numu_bar')))
            
        int_eps_val = np.trapz(np.array(integral_eps) * eps_grid, x=np.log(eps_grid))
        integral_x.append(Np_val * int_eps_val)

    return np.trapz(np.array(integral_x), x=np.log(x_grid))

def suppression_factor(E_nu_prime_erg, B_prime, c, sigma_T):
    E_pi_prime, E_mu_prime = 4.0 * E_nu_prime_erg, 3.0 * E_nu_prime_erg
    t_dec_pi = (E_pi_prime / m_pi_erg) * tau_0_pi
    t_dec_mu = (E_mu_prime / m_mu_erg) * tau_0_mu
    
    rate_sync_pi = t_sync_rate(m_e, B_prime, E_pi_prime, c, sigma_T) * (m_e / (m_pi_erg/c**2))**4
    rate_sync_mu = t_sync_rate(m_e, B_prime, E_mu_prime, c, sigma_T) * (m_e / (m_mu_erg/c**2))**4
    
    t_sync_pi = 1.0 / rate_sync_pi if rate_sync_pi > 0 else np.inf
    t_sync_mu = 1.0 / rate_sync_mu if rate_sync_mu > 0 else np.inf
    
    S_pi = 1.0 / (1.0 + t_dec_pi / t_sync_pi)
    S_mu = 1.0 / (1.0 + t_dec_mu / t_sync_mu)
    return 0.5 * S_pi + 0.5 * (S_pi * S_mu)

d_CenA_cm = 3.8 * 3.086e24 

def calculate_neutrino_flux_earth_pg_exact(E_nu_obs_GeV, base_p, N_pro_interp, n_ph_interp):
    E_nu_obs_erg = E_nu_obs_GeV * gev_to_erg
    D_factor = 1.0 / (base_p['Gamma_b'] * (1.0 - base_p['beta'] * np.cos(base_p['theta_rad'])))
    E_nu_prime_erg = E_nu_obs_erg / D_factor 

    def volume_integrand(z_cm):
        Q_prime = calculate_Q_nu_pg_exact(E_nu_prime_erg, z_cm, N_pro_interp, n_ph_interp, base_p)
        B_z = calculate_B_field(base_p['B_0_Gauss'], z_cm, base_p['Z_0_cm'], base_p['m_idx'])
        Gamma_z = base_p['Gamma_b'] # Aproximação constante de Gamma na integração
        B_prime_z = calculate_B_prime(B_z, Gamma_z)
        
        Q_prime *= suppression_factor(E_nu_prime_erg, B_prime_z, c, sigma_T)
        area_secao = np.pi * (z_cm * np.tan(np.radians(base_p['xi_j_deg'])))**2
        return Q_prime * area_secao

    z_grid_total = np.linspace(base_p['Z_acc_cm'], base_p['Z_acc_cm'] + base_p['delta_z_cm'], 50)
    integrand_vals = np.array([volume_integrand(z) for z in z_grid_total])
    
    flux_differential = (D_factor / (d_CenA_cm**2)) * np.trapz(integrand_vals, x=z_grid_total) 
    return ((E_nu_obs_erg**2) * flux_differential) / gev_to_erg

def Q_nu_from_pion_prime(E_nu_erg, z_cm, N_pi_interp_2d, E_max_pro_erg):
    lower_limit = E_nu_erg / (1.0 - r_pi)
    if lower_limit >= E_max_pro_erg: return 0.0
    E_pi_grid = np.logspace(np.log10(lower_limit), np.log10(E_max_pro_erg), 100)
    
    def calc_integrand(E_pi_erg):
        if E_nu_erg / E_pi_erg > (1.0 - r_pi): return 0.0
        return (N_pi_interp_2d(E_pi_erg, z_cm) / T_decay_pion(E_pi_erg)) * (1.0 / (E_pi_erg * (1.0 - r_pi)))

    vals = np.array([calc_integrand(E) for E in E_pi_grid])
    return np.trapz(vals * E_pi_grid, x=np.log(E_pi_grid))

def Q_nu_from_muon_prime(E_nu_erg, z_cm, N_mu_L_interp, N_mu_R_interp, E_max_pro_erg):
    if E_nu_erg >= E_max_pro_erg: return 0.0
    E_mu_grid = np.logspace(np.log10(E_nu_erg), np.log10(E_max_pro_erg), 100)
    
    def calc_integrand(E_mu_erg):
        x = E_nu_erg / E_mu_erg
        if x >= 1.0: return 0.0
        N_mu_tot = N_mu_L_interp(E_mu_erg, z_cm) + N_mu_R_interp(E_mu_erg, z_cm)
        if N_mu_tot <= 0: return 0.0
        return (1.0 / E_mu_erg) * (1.0 / T_decay_muon(E_mu_erg)) * N_mu_tot * ((5.0/3.0) - 3.0*(x**2) + (4.0/3.0)*(x**3))

    vals = np.array([calc_integrand(E) for E in E_mu_grid])
    return np.trapz(vals * E_mu_grid, x=np.log(E_mu_grid))

def calculate_neutrino_flux_earth_pp(E_nu_obs_GeV, base_p, N_pi_interp, N_mu_L_interp, N_mu_R_interp):
    E_nu_obs_erg = E_nu_obs_GeV * gev_to_erg
    D_factor = 1.0 / (base_p['Gamma_b'] * (1.0 - base_p['beta'] * np.cos(base_p['theta_rad'])))
    E_nu_prime_erg = E_nu_obs_erg / D_factor

    def volume_integrand(z_cm):
        q_pi = Q_nu_from_pion_prime(E_nu_prime_erg, z_cm, N_pi_interp, base_p['E_max_pro_erg'])
        q_mu = Q_nu_from_muon_prime(E_nu_prime_erg, z_cm, N_mu_L_interp, N_mu_R_interp, base_p['E_max_pro_erg'])
        area_secao = np.pi * (z_cm * np.tan(np.radians(base_p['xi_j_deg'])))**2
        return (q_pi + q_mu) * area_secao

    z_grid_total = np.logspace(np.log10(base_p['Z_acc_cm']), np.log10(100.0 * base_p['Z_acc_cm']), 100)
    integrand_vals = np.array([volume_integrand(z) for z in z_grid_total])
    
    flux_differential = (D_factor / ( d_CenA_cm**2)) * np.trapz(integrand_vals, x=z_grid_total) 
    return ((E_nu_obs_erg**2) * flux_differential * (2.0/3.0)) / gev_to_erg # 2/3 de prob. de oscilação


def run_pipeline(source_file, output_dir):
    """
    Roda o pipeline completo (parâmetros base -> elétrons -> prótons) para um
    único arquivo de fontes, salvando tudo dentro de output_dir.

    Retorna um dicionário: { 'NomeDaFonte': base_p (com resultados), ... }
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"=== PIPELINE: lendo '{source_file}' -> salvando em '{output_dir}/' ===")
    print(f"{'='*70}")

    source_list = []
    try:
        with open(source_file, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                source_list.append(row)
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{source_file}' não encontrado. Pulando esta execução.")
        return {}

    if not source_list:
        print(f"Nenhuma fonte encontrada em '{source_file}'.")
        return {}

    all_base_parameters = {}
    parameters_path = os.path.join(output_dir, "all_calculated_parameters.csv")

    # --- 1) Parâmetros base + Elétrons ---
    print(f"=== [{output_dir}] Calculando parâmetros e elétrons ({len(source_list)} fontes) ===")
    global_photon_funcs = {}

    with open(parameters_path, mode='w', newline='') as param_file:
        writer = csv.writer(param_file)

        for idx, params in enumerate(source_list):
            source_name = params['name']
            base_p = get_base_params(params)
            all_base_parameters[source_name] = base_p

            if idx == 0:
                header = ['Source_Name'] + list(base_p.keys())
                writer.writerow(header)
            writer.writerow([source_name] + list(base_p.values()))

            photon_func = process_and_save_electrons(source_name, base_p, output_dir=output_dir)
            global_photon_funcs[source_name] = photon_func

    print(f"=== [{output_dir}] Parâmetros salvos em '{parameters_path}' ===\n")

    # --- 2) Prótons (depende da função de fótons calculada acima) ---
    print(f"=== [{output_dir}] Calculando prótons ({len(source_list)} fontes) ===")

    for source_name, base_p in all_base_parameters.items():
        n_ph_func = global_photon_funcs.get(source_name)
        if n_ph_func is not None:
            process_and_save_protons(source_name, base_p, n_ph_func, output_dir=output_dir)
        else:
            print(f"Warning: Photon function for {source_name} not found. Skipping protons.")

    print(f"=== [{output_dir}] PIPELINE CONCLUÍDO para '{source_file}' ===\n")

    return all_base_parameters


if __name__ == "__main__":
    import os
    
    # Define a lista de tarefas
    runs = [
        ("source.txt", "data"),
        ("source2.txt", "data2"),   
        ("source3.txt", "data3"),   
    ]

    # Pega o ID da tarefa atual passado pelo SLURM (padrão é 0 para testes locais)
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    # Verifica se o ID é válido dentro da nossa lista
    if task_id < len(runs):
        source_file, output_dir = runs[task_id]
        
        print(f"\n=== INICIANDO TAREFA SLURM {task_id} ===")
        print(f"Processando arquivo: {source_file} -> Pasta: {output_dir}")
        
        # Executa o pipeline apenas para o arquivo deste job específico
        resultados = run_pipeline(source_file, output_dir)
        
        print(f"=== TAREFA {task_id} FINALIZADA com {len(resultados)} fontes ===")
    else:
        print(f"Erro: TASK_ID {task_id} está fora dos limites da lista de execuções.")