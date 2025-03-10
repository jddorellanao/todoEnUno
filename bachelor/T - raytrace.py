import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp

def rhsf(l, r, slowness, dsdx, dsdz, xaxis, zaxis, dx, dz):
    """RHS of raytracing ODE 
    
    Parameters
    ----------
    r : dependent variable containing (x, z, px, pz, t)
    l : indipendent variable l
    slowness : slowness 2d model
    dsdx : horizontal derivative of slowness 2d model
    dsdz : vertical derivative of slowness 2d model
    xaxis : horizontal axis
    zaxis : vertical axis
    dx : horizontal spacing
    dz : vertical spacing

    Returns
    -------
    drdt : RHS evaluation
    
    """
    m, n = slowness.shape
    # extract the different terms of the solution
    x = r[0]
    z = r[1]
    px = r[2]
    pz = r[3]
    drdt = np.zeros(len(r))

    # identify current position of the ray in the model
    xx = (x - xaxis[0]) // dx
    zz = (z - zaxis[0]) // dz
    xx = min([xx, n-1])
    xx = max([xx, 1])
    zz = min([zz, m-1])
    zz = max([zz, 1]) 

    # extract s, ds/dx, ds/dz at current position (nearest-neighbour interpolation)
    s = slowness[round(zz), round(xx)]
    dsdx = dsdx[round(zz), round(xx)]
    dsdz = dsdz[round(zz), round(xx)]
    
    # evaluate RHS
    drdt[0] = px/s
    drdt[1] = pz/s
    drdt[2] = dsdx
    drdt[3] = dsdz
    drdt[4] = s
    return drdt

def event_left(l, r, slowness, dsdx, dsdz, xaxis, zaxis, dx, dz):
    return r[0]-xaxis[0]
def event_right(l, r, slowness, dsdx, dsdz, xaxis, zaxis, dx, dz):
    return xaxis[-1]-r[0]
def event_top(l, r, slowness, dsdx, dsdz, xaxis, zaxis, dx, dz):
    return r[1]-zaxis[0]
def event_bottom(l, r, slowness, dsdx, dsdz, xaxis, zaxis, dx, dz):
    return zaxis[-1]-r[1]

event_left.terminal = True # set to True to trigger termination as soon as the condition is met
event_left.direction = -1 # set to -1 if wa went to stop when going from positive to negative outputs of event
event_right.terminal = True # set to True to trigger termination as soon as the condition is met
event_right.direction = -1 # set to -1 if wa went to stop when going from positive to negative outputs of event
event_top.terminal = True # set to True to trigger termination as soon as the condition is met
event_top.direction = -1 # set to -1 if wa went to stop when going from positive to negative outputs of event
event_bottom.terminal = True # set to True to trigger termination as soon as the condition is met
event_bottom.direction = -1 # set to -1 if wa went to stop when going from positive to negative outputs of event

def raytrace(vel, xaxis, zaxis, dx, dz, lstep, source, thetas):
    """Raytracing for multiple rays defined by the initial conditions (source, thetas)
    
    Parameters
    ----------
    vel : np.ndarray
        2D Velocity model (nz x nx)
    xaxis : np.ndarray
        Horizonal axis 
    zaxis : np.ndarray
        Vertical axis 
    dx : float
        Horizonal spacing 
    dz : float
        Vertical spacing 
    lstep : np.ndarray
        Ray lenght axis
    source : tuple
        Source location
    thetas : tuple
        Take-off angles
    
    """
    # Slowness and its spatial derivatives
    slowness = 1./vel;
    [dsdz, dsdx] = np.gradient(slowness, dz, dx)

    df = pd.DataFrame(columns = ['Source', 'Theta', 'rx', 'rz'])

    for theta in thetas:
        # Initial condition
        r0=[source[0], source[1], 
            sin(theta * np.pi / 180) / vel[izs, ixs],
            cos(theta * np.pi / 180) / vel[izs, ixs], 0]

        # Solve ODE
        sol = solve_ivp(rhsf, [lstep[0], lstep[-1]], r0, t_eval=lstep, 
                        args=(slowness, dsdx, dsdz, x, z, dx, dz), events=[event_right, event_left,
                                                                           event_top, event_bottom])
        r = sol['y'].T
        
        # Display ray making sure we only plot the part of the ray that is inside the model
        zeros  = np.where(r[1:, 1] <= 0)[0]
        maxs = np.where(r[:, 1] >= max(z))[0]
        
        # Coordenadas del rayo y se guardan
        rx, ry = r[:,0]/1000, r[:,1]/1000

        for a in range(rx.size):

            append = pd.Series(
                {
                    'Source':source,
                    'Theta':theta,
                    'rx':rx[a],
                    'rz':ry[a]
                }
            )
            df = pd.concat([df, append.to_frame().T], ignore_index=True)

    return df