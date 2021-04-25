import numpy as np

from tenpy.networks.site import Site, SpinHalfFermionSite, SpinHalfSite, GroupedSite, SpinSite
from tenpy.tools.misc import to_iterable, to_iterable_of_len, inverse_permutation
from tenpy.networks.mps import MPS  # only to check boundary conditions

from tenpy.models.lattice import Lattice, _parse_sites, Honeycomb
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.tools.params import get_parameter

import tenpy
from tenpy.models import lattice
from tenpy.algorithms import dmrg

class KitaevHoneycombModel(CouplingMPOModel):
    """
        Defining the MPO for Kitaev Honeycomb
    """
    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        # conserve = get_parameter(model_params, 'conserve', None, self.name)
        conserve = model_params.get('conserve', None)
        S = model_params.get('S', 0.5) # by default spin one half
        fs = SpinSite(S=S, conserve=conserve)
        #! Add another real operator to make sure that the whole problem can be solved within real space
        iSy = np.real(1.j*fs.get_op('Sy').to_ndarray())
        fs.add_op('iSy', iSy)
        return [fs, fs]

    def init_lattice(self, model_params):
        Lx = model_params.get('Lx', 4) # 2 is the least possible number for L to be a Kitaev ladder we want, and 4 is more secured (I hope so)
        Ly = model_params.get('Ly', 3)
        
        gs = self.init_sites(model_params)
        model_params.pop("Lx")
        model_params.pop("Ly")


        order = model_params.get('order', 'Cstyle')
        bc = model_params.get('bc', 'periodic')
        shift = model_params.get('shift', 0)
        bc_MPS=model_params.get('bc_MPS', 'finite')
        lattice_params = dict(
            order=order,
            bc=[bc, shift],
            bc_MPS=bc_MPS,
            basis=None,
            positions=None,
            nearest_neighbors=None,
            next_nearest_neighbors=None,
            next_next_nearest_neighbors=None,
            pairs={},
        )

#         lat = KitaevLadderSnakeCompact(L, gs, **lattice_params)
        lat = Honeycomb(Lx, Ly, sites=gs, **lattice_params)
        return lat

    def init_terms(self, model_params):
        # Jx = get_parameter(model_params, 'Jx', 1., self.name, True)
        # Jy = get_parameter(model_params, 'Jy', 1., self.name, True)
        # Jz = get_parameter(model_params, 'Jz', 1., self.name, True)
        Jx = model_params.get('Jx', 1.)
        Jy = model_params.get('Jy', 1.)
        Jz = model_params.get('Jz', 1.)

        u1, u2, dx = self.lat.pairs['nearest_neighbors'][0]
        self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)
        u1, u2, dx = self.lat.pairs['nearest_neighbors'][1]
        self.add_coupling(-Jy, u1, 'iSy', u2, 'iSy', dx)
        u1, u2, dx = self.lat.pairs['nearest_neighbors'][2]
        self.add_coupling(Jx, u1, 'Sx', u2, 'Sx', dx)

def run_atomic(
    # model parameters
    chi=30, 
    Jx=1., 
    Jy=1., 
    Jz=1., 
    S=.5, 
    Lx=4, 
    Ly=3, 
    bc='periodic', 
    shift=0,
    bc_MPS='infinite', 
    conserve='parity',
    order='Cstyle',
    dtype=float,
    ####### dmrg parameters #######
    initial_psi=None, # input psi
    initial='random', 
    max_E_err=1.e-6, 
    max_S_err=1.e-4, 
    max_sweeps=200, 
    N_sweeps_check=10, 
    canonicalized=True, 
    mixer=True,
    mixer_params={
        'amplitude': 1.e-4,
        'decay': 1.2,
        'disable_after': 50
    },
    trunc_params={
        'chi_max': 4,
        'svd_min': 1.e-10,
    },
    # control for the verbose output
    verbose=1, 
):
    """ 
        The fundamental function for running DMRG
    """

    #######################
    # set the paramters for model initialization
    model_params = dict(
        conserve=conserve, 
        Jx=Jx, 
        Jy=Jy, 
        Jz=Jz, 
        Lx=Lx, 
        Ly=Ly, 
        S=S,
        verbose=verbose,
        bc=bc,
        bc_MPS=bc_MPS,
        order=order,
        dtype=dtype,
        )

    L = Lx*Ly # the number of unicells
    # initialize the model
    M = KitaevHoneycombModel(model_params)
    prod_state = ["up"] * (2*L) 
    psi = MPS.from_product_state(
        M.lat.mps_sites(), 
        prod_state, 
        bc=M.lat.bc_MPS,
    )
    
    #######################
    # set the parameters for the dmrg routine
    dmrg_params = {
        'start_env': 10,
        'mixer': mixer,
        'mixer_params': mixer_params,
        
        'trunc_params': trunc_params,
        'max_E_err': max_E_err,
        'max_S_err': max_S_err,
        'max_sweeps': max_sweeps,
        'N_sweeps_check': N_sweeps_check,
        'verbose': verbose,
    }
    #######################
    
    if verbose:
        print("\n")
        print("=" * 80)
        print("="*30 + "START" + "="*30)
        print("=" * 80)
        print("Chi = ", chi, '\n')

    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    eng.reset_stats()
    eng.trunc_params['chi_max'] = chi
    info = eng.run()

    if canonicalized:
        psi.canonical_form()
        if verbose:
            print("Before the canonicalization:")
            print("Bond dim = ", psi.chi)

            print("Canonicalizing...")
            psi_before = psi.copy()


        if verbose:
            ov = psi.overlap(psi_before, charge_sector=0)
            print("The norm is: ",psi.norm)
            print("The overlap is: ", ov)
            print("After the canonicalization:")
            print("Bond dim = ", psi.chi)

        print("Computing properties")

    energy=info[0]

    if verbose:
        print("Optimizing")

    tenpy.tools.optimization.optimize(3)

    if verbose:
        print("Loop for chi=%d done." % chi)
        print("=" * 80)
        print("="*30 + " END " + "="*30)
        print("=" * 80)
        
    # the wave function, the ground-state energy, and the DMRG engine are all that we need
    result = dict(
        psi=psi.copy(),
        energy=energy,
        sweeps_stat=eng.sweep_stats.copy(),
        parameters=dict(
            # model parameters
            chi=chi,
            Jx=Jx, 
            Jy=Jy, 
            Jz=Jz, 
            L=L, 
            # dmrg parameters
            initial_psi=initial_psi,
            initial=initial,
            max_E_err=max_E_err,
            max_S_err=max_S_err,
            max_sweeps=max_sweeps,
        )
    )
    return result
