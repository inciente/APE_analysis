import numpy as np; import xarray as xr; import pandas as pd; 
import scipy.stats as stats; import sys, os; 
from datetime import datetime, timedelta; 

'''
Create a set of functions that iterates through ocean models and does the following:
- Load dataset
- Define and perform spatial subsetting
- Load corresponding refernce density time series
- Using ds and ref_rho:
   1. Instantiate Tailleux for (-3,3N)
       a) compute Ea
       b) Compute 2D advection of Ea
       c) Get MLD from taill.ds['HMXL']
       d) Integrate horizontal convergence of Ea
       e) Return and save as timeseries (units of watts)

    2. Instantiate Tailleux at 10 and 15 N.
       a) Compute Ea
       b) Use taill.ds['VVEL'] / 100 to do v* Ea
       c) Area integral in inflow area
       d) Return and save as time series (units of W / m2 )

    3. Extract w at 100 m over (-3,3N) and (215 to 260 E).
       a) easy peasy. take average and return as timeseries


'''

# Functions necessary to build this

def prepare_dataset( model ):
    ds = pull_combined_json( model ); 
    ds = ds.sel( lon = slice( 100 , 290 ) );
    # block out other basins
    ds = KE_tools.pacific_only( ds )
    ds = euc_tools.vertical_chunk( ds , 350 )
    # make it learn its own dimensions
    ds['dx'], ds['dy'], ds['dz'] = grid_sizes( ds )
    # get reference density profiles
    ds = add_ref_rho_to_ds( model , ds ); 
    return ds

def find_ref_dens( model ):
    filename = 'reference_profiles/' + model.config + '_ENS' + str( model.ensnum ).zfill( 2 ) \
               + '_RHO_REF_PACIFIC_lat35_coarse.nc'
    ref_rho = xr.open_dataset( filename )
    return ref_rho

def add_ref_rho_to_ds( model , ds ):
    # Get reference density profile and add it to the variables in ds
    ref_rho = find_ref_dens( model )
    ref_rho = ref_rho.interp( time = ds['time'] ).interp( z_t = ds['z_t'] )
    ds['ref_rho'] = ref_rho['ref_rho'];
    return ds

def grid_sizes( ds ):
    dx , dy = thermo.get_dx_dy( ds )
    dz = thermo.get_dz( ds )
    return dx, dy, dz

# ------------------------------------------------
# Spatial subsetting tools
eq_subset = lambda ds : ds.sel( { 'lon' : slice( 150, 205 ) , 'lat' : slice( -3, 3 ) } )

ten_subset = lambda ds : ds.sel( {'lon':slice(150,205) } ).sel( lat = [10, 15] , method = 'nearest' )

ep_subset = lambda ds : ds.sel( {'lon':slice(215,260) , 'lat':slice(-3,3) } )

# -------------------------------------------------
# Functions for energetics
def instantiate_locations( ds ):
    # Take ds and create all subdatasets / energy instances necessary
    # ds must already contain reference profile
    # ds must be a snapshot (no time dimension)
    ds_eq = eq_subset( ds )
    taill_eq = KE_tools.Tailleux( ds_eq, ds['ref_rho'] )

    ds_ten = ten_subset( ds ) 
    taill_ten = KE_tools.Tailleux( ds_ten , ds['ref_rho'] )

    ds_ep = ep_subset( ds )

    return taill_eq, taill_ten, ds_ep 

def equatorial_energy( taill ):
    # Take in instance of tailleux and perform what's needed
    Ea = taill.boussinesq_PI2()
    Ea_conv = taill.advective_term( Ea , three_d = False )
    # mask for convergence regions
    conv_mask = ( Ea_conv < 0 ) * ( Ea_conv['z_t'] > taill.ds['HMXL'] / 100 )
    # integrate the result
    integral = ( taill.ds['dx'] * taill.ds['dy'] * taill.ds['dz'] \
                * Ea_conv.where( conv_mask ) ).sum( ['lon','lat','z_t'] )
    return integral

def eastern_pac_indicators( ds ):
    # Mean upwelling
    w = ( ds['WVEL'].sel( z_w_top = 100 , method = 'nearest' ) / 100).mean(['lon','lat'])
    tau = ( ds['TAUX'] ).mean( ['lon','lat'] );
    nures = xr.Dataset()
    nures['w'] = w; nures['tau'] = tau; 
    return nures


def meridional_transport( taill ):
    # Take instance of tailleux and perform what's needed 
    Ea = taill.boussinesq_PI2()
    vEa = taill.ds['VVEL'] / 100 * Ea
    integral = ( taill.ds['dz'] * taill.ds['dx'] * vEa )
    integral = integral.sel( z_t = slice( 75, 200 ) ).sum( ['lon','z_t'] )
    return integral

def get_energetics( data_snapshot , res_dict ):
    # Receives snapshot of datasets and performs analyses
    taill_eq, taill_ten, ds_ep = instantiate_locations( data_snapshot )

    # Call analysis functions and append results to dict lists
    res_dict['Ea_conv'].append( equatorial_energy( taill_eq ) )
    res_dict['vEa'].append( meridional_transport( taill_ten ) )
    res_dict['ep_w'].append( eastern_pac_indicators( ds_ep ) )
    return res_dict

# ------------------------------------------
# Perform operation and save output
def results_filename( model ):
    fln = 'ape_timeseries/energetics_' + model.config + '_ENS' + str( model.ensnum ).zfill( 2 ) + '_lat35.nc'
    return fln

def operate_on_model( model , yearly_avgs = True ):
    # Load data
    ds = prepare_dataset( model )
    print('Loaded data for ' + model.config + ' ENS' \
                         + str( model.ensnum ).zfill(2) )
    
    if yearly_avgs:
        ds = ds.groupby( 'time.year' ).mean();
        time_dim = 'year'; tstep = 15;
    else:
        time_dim = 'time'; tstep = 50;
        
    # Create dictionary to store results
    results = { 'Ea_conv':[] , 'ep_w':[], 'vEa':[] }

    # Iterate through time
    for tt in range( len( ds[ time_dim ] ) - 1 ):
        if np.mod( tt , tstep ) == 0:
            print( 'Running timestep ' + str( tt ) )
        data_snapshot = ds.isel( { time_dim : tt } )
        ref_rho = data_snapshot['ref_rho']

        # Perform analyses and save results
        results = get_energetics( data_snapshot , results )

    # Concatenate items in results
    results_xr = xr.concat( results['ep_w'] , dim = time_dim ).persist(); # this is already a dataset
    for var in ['Ea_conv','vEa']:
        # store remaining results in the same dataset
        results_xr[var] = xr.concat( results[var] , dim = time_dim ).persist()
    
    # Save as netcdf
    results_xr.to_netcdf( results_filename( model ) );
    
    return results_xr



