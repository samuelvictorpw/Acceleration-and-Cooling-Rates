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

# ==========================================================================
# NUCLEO ACELERADO: OXIGENIO (O-16)
# ==========================================================================
Z_O = 8                        # numero atomico (carga em unidades de e)
A_O = 16                       # numero de massa
m_O = A_O * m_p                # massa do nucleo (aprox., ignora defeito de massa ~0.8%)
q_O = Z_O * e                  # carga eletrica do nucleo

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

# --- ACELERAÇÃO: agora depende da carga q da partícula (q = Z*e) ---
def t_acc_rate(n, c, q, B_prime, E_prime):
    """q = carga da partícula acelerada (para O: q_O = Z_O*e; para eletron/proton: e)."""
    return (n * c * q * B_prime) / E_prime

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

# ==========================================================================
# IC e Fótons (inalterado — depende só de elétrons/fótons, não da espécie
# de núcleo acelerado)
# ==========================================================================
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
    return np.trapezoid(vals * q_grid, x=np.log(q_grid))

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
    int_ele = np.trapezoid(integrand_2d, x=E_ele_grid, axis=1)
    res = np.trapezoid(int_ele, x=E_ph_grid)
    
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
    return pre_factor * np.trapezoid(vals * E_ele_grid, x=np.log(E_ele_grid))

def calculate_n_ph_prime(E_ph_prime, r_j_cm, K_e, s, E_max_ele, m_e, B_prime, c, sigma_T, t_ad_val, n_c_prime, h, e, E_min_ele_erg, Ne_interp_func):
    lower_lim = m_e * c**2
    upper_lim = E_max_ele * 10.0 
    
    E_ele_grid = np.logspace(np.log10(lower_lim), np.log10(upper_lim), 150)
    vals = np.array([emissivity_integrand(E_ele, E_ph_prime, K_e, s, E_max_ele, m_e, B_prime, c, sigma_T, t_ad_val, n_c_prime, h, e, Ne_interp_func) for E_ele in E_ele_grid])
    
    eps_syn = 4 * np.pi * np.trapezoid(vals * E_ele_grid, x=np.log(E_ele_grid))
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
    
    return const_term * np.trapezoid(vals * E_ph_grid, x=np.log(E_ph_grid))

def generate_IC_rate_table(E_GeV_array, rate_ic_num_array, Gamma_b, beta, m_e, c, thetaR):
    print("   -> [IC] Generating IC rate table...")
    E_arr = np.array([calculate_E_prime(Gamma_b, E_g * gev_to_erg, beta, m_e, c, thetaR) for E_g in E_GeV_array])
    interp_log = interp1d(np.log10(E_arr), np.log10(np.maximum(rate_ic_num_array, 1e-100)), kind='linear', fill_value=-100.0, bounds_error=False)
    
    def ic_rate_interp_func(E_prime_erg):
        if E_prime_erg <= 0: return 0.0
        log_rate = interp_log(np.log10(E_prime_erg))
        return 10**log_rate if log_rate > -90 else 0.0
    return ic_rate_interp_func

# ==========================================================================
# Transporte
# ==========================================================================
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

def b_total_z_oxygen(E_prime, z, p, n_ph_func=None):
    """Taxa de perda de energia (dE/dt * E, na verdade dE/dt em erg/s) para
    núcleos de oxigênio (Z_O=8, A_O=16), substituindo b_total_z_proton.

    Componentes:
      - Síncrotron: escala com q^4/m^2 -> fator (Z_O**4)*(m_e/m_O)**4 sobre a
        taxa de referência do elétron.
      - Adiabática: igual (depende só da geometria do jato).
      - O-p inelástica (analog de pp): seção de choque efetiva escalada por
        A_O**0.9 (modelo de superposição/wounded-nucleon), limiar em energia
        por núcleon.
      - Fotoprodução de píon (p-gamma -> O-gamma): escalada por A_O**0.9,
        gamma de Lorentz calculado com m_O.
      - Fotodesintegração (GDR): processo NOVO, dominante para núcleos e
        ausente na versão para prótons. Ver fotodesintegracao_rate().
    """
    Bz   = calculate_B_field(p['B0'], z, p['z0'], p['m_idx'])
    Bpr  = calculate_B_prime(Bz, p['Gamma_b'])
    nc_p = p['n_c_zacc'] * (p['z_acc'] / z)**2
    
    loss = t_sync_rate_oxygen(Bpr, E_prime, c, sigma_T) * E_prime
    loss += t_adiabatic_rate(z, p['beta'], c) * E_prime
    loss += Op_rates(nc_p, c, E_prime) * E_prime
    
    if n_ph_func is not None:
        loss += t_pg_rate_oxygen(E_prime, n_ph_func, c, m_O, Z_O, A_O,
                                  E_ph_min_erg=p.get('E_ph_min_erg'),
                                  E_ph_max_erg=p.get('E_ph_max_erg')) * E_prime
        loss += photodisintegration_rate(E_prime, n_ph_func, c, m_O, A_O, Z_O) * E_prime
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

    return max(float(np.trapezoid(integrand, x=sol.t)), 0.0)

def build_N_grid(E_grid_erg, z_grid_cm, species, tp, extra_func=None, verbose=True):
    # 'oxygen' substitui 'proton' como a espécie hadrônica acelerada
    if species == 'electron':
        b_func = b_total_z_electron
    elif species == 'oxygen':
        b_func = b_total_z_oxygen
    else:
        raise ValueError(f"Espécie desconhecida: {species}")

    Ki = tp['K_e'] if species == 'electron' else tp['K_O']
    
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

# ==========================================================================
# Física Focada em Núcleos de Oxigênio (substitui a seção "Prótons")
# ==========================================================================

def t_sync_rate_oxygen(B_prime, E_prime, c, sigma_T):
    """Perda síncrotron para um núcleo de carga Z_O e massa m_O.
    P_sync ~ q^4 B'^2 gamma^2 / m^2  =>  reescala em relação ao elétron por
    (Z_O**4) * (m_e/m_O)**4 (a potência 4 em massa aparece pois a fórmula de
    referência t_sync_rate já embute e^2/m_e^2 dentro de sigma_T)."""
    return t_sync_rate(m_e, B_prime, E_prime, c, sigma_T) * (Z_O**4) * (m_e / m_O)**4

def sigma_Op_inel(E_O_GeV, A=A_O):
    """Seção de choque inelástica O-p efetiva via modelo de superposição de
    núcleons (wounded nucleon model): sigma_Ap ~ A^0.9 * sigma_pp, com o
    limiar avaliado em energia POR NUCLEON (E/A)."""
    E_p_equiv_GeV = E_O_GeV / A          # energia por núcleon
    E_th = 1.22
    if E_p_equiv_GeV <= E_th: return 0.0
    L = np.log(E_p_equiv_GeV / 1000.0)
    sigma_pp = (34.3 + 1.88*L + 0.25*L**2) * ((1.0 - (E_th/E_p_equiv_GeV)**4)**2) * 1e-27
    return (A**0.9) * sigma_pp

def Op_rates(n_c_prime, c, E_prime_erg, A=A_O):
    """Taxa de perda de energia por colisões inelásticas O + p (análogo de
    pp_rates, mas com seção de choque escalada por A^0.9 e limiar por núcleon)."""
    E_O_GeV = E_prime_erg / gev_to_erg
    sigma = sigma_Op_inel(E_O_GeV, A)
    if sigma <= 0: return 1e-30
    rate = n_c_prime * c * sigma * 0.5
    return max(rate, 1e-30)

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

def sigma_pairs(x_prime, Z=Z_O):
    """Produção de pares Bethe-Heitler no campo do núcleo: escala com Z^2
    (campo coulombiano do núcleo inteiro, vs. campo de um único próton)."""
    if x_prime < 4.0:
        eta = (x_prime - 2)/(x_prime + 2)
        series = 1 + 0.5*eta + (23/40)*(eta**2) + (37/120)*(eta**3) + (61/192)*(eta**4)
        base = 1.2135e-27 * ((x_prime - 2)/2)**3 * series
    else:
        ln_2x = np.log(2*x_prime)
        term1 = 3.1111*ln_2x - 8.0741
        term2 = (2/x_prime)**2 * (2.7101*ln_2x - (ln_2x**2) + 0.6667*(ln_2x**3) + 0.5490)
        base = 5.7938e-28 * (term1 + term2)
    return (Z**2) * base

def get_sigma_K_pion(x_prime, A=A_O):
    """Fotoprodução de píon via ressonância Delta. Escalada por A^0.9
    (superposição de núcleons) em relação ao núcleon livre."""
    if x_prime < 284: return 0.0
    K_pi = 9.78e-3 * (x_prime**0.4756)
    if 284 <= x_prime < 644: sigma_pi = 5.48e-28 * (x_prime/644)**2.60
    elif 644 <= x_prime < 978: sigma_pi = 5.48e-28 * (x_prime/644)**(-2.89)
    elif 978 <= x_prime < 1370: sigma_pi = 1.64e-28 * (x_prime/978)**1.51
    elif 1370 <= x_prime < 2387: sigma_pi = 2.73e-28 * (x_prime/1370)**(-1.20)
    else: sigma_pi = 1.4e-27 * 0.5 
    return (A**0.9) * sigma_pi * K_pi

def t_pg_rate_oxygen(E_O_erg, n_ph_func, c, m_nucleus, Z, A, E_ph_min_erg=None, E_ph_max_erg=None):
    """Fotoprodução de píon O+gamma. O gamma de Lorentz é calculado com a
    massa TOTAL do núcleo (m_nucleus = A*m_p), o que desloca o limiar em
    energia total (mas mantém o mesmo limiar em energia POR NUCLEON que o
    caso do próton)."""
    gamma_O = E_O_erg / (m_nucleus * c**2)
    lower = max((2.0 * mec2_erg) / (2.0 * gamma_O), E_ph_min_erg) if E_ph_min_erg else (2.0 * mec2_erg) / (2.0 * gamma_O)
    upper = E_ph_max_erg or 1.0e8 * eV_to_erg
    if lower >= upper: return 1e-30

    def integrando_interno(epsilon_r):
        x_prime = epsilon_r / mec2_erg
        return (get_sigma_K_pion(x_prime, A) + (sigma_pairs(x_prime, Z) * K_pairs(x_prime))) * epsilon_r

    def integrando_externo(E_ph_erg):
        n_density = get_n_ph_val(E_ph_erg, n_ph_func)
        if n_density <= 0.0: return 0.0
        epsilon_max = 2.0 * gamma_O * E_ph_erg
        if epsilon_max <= 2.0 * mec2_erg: return 0.0
        pts = [p for p in [284.0 * mec2_erg, 644.0 * mec2_erg, 978.0 * mec2_erg] if 2.0 * mec2_erg < p < epsilon_max]
        val_interna, _ = spi.quad(integrando_interno, 2.0 * mec2_erg, epsilon_max, points=pts, limit=300, epsabs=0, epsrel=1e-3)
        return (n_density / E_ph_erg**2) * val_interna

    E_ph_grid = np.logspace(np.log10(lower), np.log10(upper), 500)
    vals = np.array([integrando_externo(E_ph) for E_ph in E_ph_grid])
    rate = (c / (2.0 * gamma_O**2)) * np.trapezoid(vals * E_ph_grid, x=np.log(E_ph_grid))
    return max(rate, 1e-30)

# --------------------------------------------------------------------
# FOTODESINTEGRAÇÃO (processo NOVO — ausente na versão para prótons).
# Para núcleos, este é tipicamente o canal DOMINANTE de perda de energia
# por interação com o campo de fótons (mais importante que p-gamma).
# Implementado com o modelo clássico de ressonância dipolar gigante (GDR)
# de Lorentziana única, normalizado pela regra de soma TRK
# (Puget, Stecker & Bredekamp 1976; Stecker & Salamon 1999):
#
#   sigma(eps) = sigma_0 * (Gamma_GDR^2 * eps^2) /
#                ((eps^2 - E_GDR^2)^2 + Gamma_GDR^2 * eps^2)
#
# com sigma_0 fixado pela regra de soma TRK:
#   integral sigma(eps) d(eps) = 60 * (N*Z/A) MeV*mb
# --------------------------------------------------------------------
E_GDR_O_MeV = 24.0        # pico da GDR do O-16 (~ typico p/ nucleos leves)
Gamma_GDR_O_MeV = 8.0     # largura da GDR do O-16 (valores tabelados em
                          # bancos de dados como IAEA/RIPL; usar os
                          # experimentais reais do O-16 se disponíveis)
mb_to_cm2 = 1e-27

def _sigma_GDR_oxygen(eps_MeV, Z=Z_O, A=A_O, E0=E_GDR_O_MeV, Gamma=Gamma_GDR_O_MeV):
    """Seção de choque de fotoabsorção (GDR, Lorentziana única) em cm^2,
    eps_MeV = energia do fóton no referencial de repouso do núcleo, em MeV."""
    N = A - Z
    sigma_int_MeVmb = 60.0 * (N * Z / A)      # regra de soma TRK, MeV*mb
    sigma_0 = sigma_int_MeVmb * (2.0 / (np.pi * Gamma))   # normalização da Lorentziana
    denom = (eps_MeV**2 - E0**2)**2 + (Gamma**2) * (eps_MeV**2)
    if denom <= 0: return 0.0
    sigma_mb = sigma_0 * (Gamma**2 * eps_MeV**2) / denom
    return sigma_mb * mb_to_cm2

def photodisintegration_rate(E_O_erg, n_ph_func, c, m_nucleus, A=A_O, Z=Z_O):
    """Taxa de perda de energia por fotodesintegração (emissão de 1 núcleon
    por interação, aproximação usual: perda de energia por evento ~ E_O/A).
    Integra sobre o campo de fótons no referencial do jato, boosteado para
    o referencial de repouso do núcleo (fator gamma_O)."""
    gamma_O = E_O_erg / (m_nucleus * c**2)
    eps_min_MeV, eps_max_MeV = 5.0, 60.0          # janela em torno da GDR
    eps_min_erg = eps_min_MeV * 1e6 * eV_to_erg
    eps_max_erg = eps_max_MeV * 1e6 * eV_to_erg

    E_ph_min_lab = eps_min_erg / (2.0 * gamma_O)
    E_ph_max_lab = eps_max_erg / (2.0 * gamma_O)
    if E_ph_min_lab >= E_ph_max_lab: return 1e-30

    def integrand(E_ph_lab):
        n_density = get_n_ph_val(E_ph_lab, n_ph_func)
        if n_density <= 0.0: return 0.0
        eps_rest_MeV = 2.0 * gamma_O * E_ph_lab / (1e6 * eV_to_erg)
        sigma = _sigma_GDR_oxygen(eps_rest_MeV, Z, A)
        return n_density * sigma

    E_ph_grid = np.logspace(np.log10(E_ph_min_lab), np.log10(E_ph_max_lab), 100)
    vals = np.array([integrand(E) for E in E_ph_grid])
    rate_interaction = (c / (2.0 * gamma_O**2)) * np.trapezoid(vals * E_ph_grid**2, x=np.log(E_ph_grid))
    # fração de energia perdida por interação ~ 1/A (emissão de 1 nucleon)
    dE_loss_rate = rate_interaction * (E_O_erg / A)
    return max(dE_loss_rate / E_O_erg, 1e-30) if E_O_erg > 0 else 1e-30

def t_shear_rate(Gamma_j, beta_j, delta_r, B_prime, E_prime, q, c):
    delta_u = beta_j * c
    r_g = E_prime / (q * B_prime)
    return 1.0 / np.maximum((3.0 * (delta_r**2) * c) / ((Gamma_j**4) * (delta_u**2) * r_g), r_g / c)

def tau_escape(E_erg, q, B_prime, c, delta_r):
    r_g = E_erg / (q * B_prime)
    return (1.5 * delta_r**2) / (r_g*c)

# ==========================================================================
# Setup de Parâmetros Globais
# ==========================================================================
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
        'Delta_V_vol': calculate_Delta_V(r_j_val_z, delta_z_cm), 'K_e_norm': K_e_val,
        'Z_0_cm': Z_0_rg * Rg, 'xi_j_deg': float(params['xi_j'])
    }

def build_transport_params(base_p, K_e, K_O, E_max_ele, E_max_O, E_ph_min_erg=None, E_ph_max_erg=None):
    return {
        'Gamma_b': base_p['Gamma_b'], 'beta': base_p['beta'], 'v_b': base_p['v_b_cm_s'],
        'z0': base_p['z0'], 'z_acc': base_p['Z_acc_cm'], 'z_end': base_p['Z_acc_cm'] + base_p['delta_z_cm'],
        'B0': base_p['B_0_Gauss'], 'm_idx': base_p['m_idx'], 'n_c_zacc': base_p['n_c_prime'],
        's': base_p['s'], 'K_e': K_e, 'K_O': K_O, 'E_max': E_max_ele,
        'E_ph_min_erg': E_ph_min_erg, 'E_ph_max_erg': E_ph_max_erg
    }

# ==========================================================================
# Simulação: Elétrons (inalterado — o(a acelera(o) primario continua sendo
# elétrons, então esta função não muda com a espécie hadrônica)
# ==========================================================================
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

# ==========================================================================
# Simulação: Núcleos de Oxigênio (substitui process_and_save_protons)
# ==========================================================================
def process_and_save_oxygen(source_name, base_p, n_ph_func_approx1, output_dir="data"):
    start_time = time.time()
    print(f"[{source_name}] Calculating Oxygen nuclei (Z={Z_O}, A={A_O})")
    
    z_grid = np.logspace(np.log10(base_p['Z_acc_cm'] * 1.001), np.log10(100.0 * base_p['Z_acc_cm']), 20)
    E_max_O = base_p.get('E_max_O_erg', A_O * 2e9 * gev_to_erg)   # rigidez max ~ igual à do proton -> energia total escala com A
    # Simplificação equivalente à original: normaliza K_O a partir de uma
    # fração 'a' da luminosidade injetada em elétrons.
    K_O_val = calculate_Ke(base_p['Delta_V_vol'] * base_p['a'] * base_p['K_e_norm'], base_p['Delta_V_vol'], base_p['s'], A_O * 1.22 * gev_to_erg, E_max_O)
    
    E_grid_O = np.logspace(np.log10(A_O * 1.22 * gev_to_erg), np.log10(E_max_O * 20), 20)
    E_ph_min, E_ph_max = (10.0 ** base_p['PH_Ei']) * eV_to_erg, (10.0 ** base_p['PH_Ef']) * eV_to_erg
    tp_O = build_transport_params(base_p, 0, K_O_val, E_max_O, E_max_O, E_ph_min, E_ph_max)

    print(f"   [{source_name}] -> Transporte de oxigênio (sync + adiabático + O-p + p-gamma + fotodesintegração)...")
    N_grid_O, N_O_interp_2d, _ = build_N_grid(E_grid_O, z_grid, 'oxygen', tp_O, n_ph_func_approx1, True)
    
    base_p['N_O_interp_2d'] = N_O_interp_2d
    
    np.save(f"{output_dir}/{source_name}_oxygen_N_grid.npy", N_grid_O)
    np.save(f"{output_dir}/{source_name}_oxygen_E_grid.npy", E_grid_O)
    np.save(f"{output_dir}/{source_name}_oxygen_z_grid.npy", z_grid)

    E_GeV_O = np.logspace(0, 21, 500)   # energia TOTAL do núcleo (GeV)
    E_prime_arr_O = np.array([calculate_E_prime(base_p['Gamma_b'], E_g*gev_to_erg, base_p['beta'], m_O, c, base_p['theta_rad']) for E_g in E_GeV_O])
    
    rate_acc_O = t_acc_rate(base_p['n'], c, q_O, base_p['B_prime_Gauss'], E_prime_arr_O)
    rate_acc_O_2 = t_acc_rate(base_p['n_2'], c, q_O, base_p['B_prime_Gauss'], E_prime_arr_O)
    rate_sync_O = t_sync_rate_oxygen(base_p['B_prime_Gauss'], E_prime_arr_O, c, sigma_T)
    rate_Op_O = Op_rates(base_p['n_c_prime'], c, E_prime_arr_O)
    rate_shear_O = t_shear_rate(base_p['Gamma_b'], base_p['beta'], base_p['r_j_z0_cm'], base_p['B_prime_Gauss'], E_prime_arr_O, q_O, c)
    rate_esc_O = tau_escape(E_prime_arr_O, q_O, base_p['B_prime_Gauss'], c, base_p['r_j_z0_cm'])
    rate_pg_O = np.array([t_pg_rate_oxygen(Ep, n_ph_func_approx1, c, m_O, Z_O, A_O, E_ph_min, E_ph_max) for Ep in E_prime_arr_O])
    rate_photodis_O = np.array([photodisintegration_rate(Ep, n_ph_func_approx1, c, m_O, A_O, Z_O) for Ep in E_prime_arr_O])

    with open(f"{output_dir}/{source_name}_oxygen.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['E_prime_GeV_total', 'rate_acc', 'rate_acc_2', 'rate_sync', 'rate_Op', 'rate_pg', 'rate_photodisintegration', 't_ad_val', 'Shear', 'ESC'])
        for i in range(len(E_GeV_O)):
            writer.writerow([E_prime_arr_O[i] / gev_to_erg, rate_acc_O[i], rate_acc_O_2[i], rate_sync_O[i], rate_Op_O[i], rate_pg_O[i], rate_photodis_O[i], base_p['t_ad_s'], rate_shear_O[i], rate_esc_O[i]])

    mins, secs = divmod(time.time() - start_time, 60)
    print(f"   === [Oxigênio] Concluído em {int(mins)}m {secs:.2f}s ===")

def run_pipeline(source_file, output_dir):
    """
    Roda o pipeline completo (parâmetros base -> elétrons -> núcleos de O)
    para um único arquivo de fontes, salvando tudo dentro de output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"=== PIPELINE (O-16): lendo '{source_file}' -> salvando em '{output_dir}/' ===")
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

    print(f"=== [{output_dir}] Calculando núcleos de oxigênio ({len(source_list)} fontes) ===")

    for source_name, base_p in all_base_parameters.items():
        n_ph_func = global_photon_funcs.get(source_name)
        if n_ph_func is not None:
            process_and_save_oxygen(source_name, base_p, n_ph_func, output_dir=output_dir)
        else:
            print(f"Warning: Photon function for {source_name} not found. Skipping oxygen.")

    print(f"=== [{output_dir}] PIPELINE CONCLUÍDO para '{source_file}' ===\n")

    return all_base_parameters


if __name__ == "__main__":
    runs = [
        ("source.txt", "data_oxygen"),
        ("source2.txt", "data2_oxygen"),
        ("source3.txt", "data3_oxygen"),
    ]

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    if task_id < len(runs):
        source_file, output_dir = runs[task_id]
        print(f"\n=== INICIANDO TAREFA SLURM {task_id} ===")
        print(f"Processando arquivo: {source_file} -> Pasta: {output_dir}")
        resultados = run_pipeline(source_file, output_dir)
        print(f"=== TAREFA {task_id} FINALIZADA com {len(resultados)} fontes ===")
    else:
        print(f"Erro: TASK_ID {task_id} está fora dos limites da lista de execuções.")