#%%
import numpy as np
import xarray as xr

# Compute EOFs, PCs and exp. Var
def varimax(
    X : np.ndarray,
    gamma : float = 1,
    max_iter : int = 1000,
    rtol : float = 1e-8
):
    '''
    adapted from https://github.com/nicrie/xeofs
    
    Perform (orthogonal) Varimax rotation.

    This implementation also works for complex numbers.

    Parameters
    ----------
    X : np.ndarray
        2D matrix to be rotated containing features as rows and modes as
        columns.
    gamma : float
        Parameter which determines the type of rotation performed: varimax (1),
        quartimax (0). Other values are possible. The default is 1.
    max_iter : int
        Number of iterations performed. The default is 1000.
    rtol : float
        Relative tolerance at which iteration process terminates.
        The default is 1e-8.

    Returns
    -------
    Xrot : np.ndarray
        Rotated matrix with same dimensions as X.
    R : array-like
        Rotation matrix of shape ``(n_rot x n_rot)``

    '''
    X = X.copy()
    n_samples, n_modes = X.shape

    if n_modes < 2:
        err_msg = 'Cannot rotate {:} modes (columns), but must be 2 or more.'
        err_msg = err_msg.format(n_modes)
        raise ValueError(err_msg)

    # Initialize rotation matrix
    R = np.eye(n_modes)

    # Normalize the matrix using square root of the sum of squares (Kaiser)
    h = np.sqrt(np.sum(X * X.conjugate(), axis=1))
    # A = np.diag(1./h) @ A

    # Add a stabilizer to avoid zero communalities
    eps = 1e-9
    X = (1. / (h + eps))[:, np.newaxis] * X

    # Seek for rotation matrix based on varimax criteria
    delta = 0.
    converged = False
    for i in range(max_iter):
        delta_old = delta
        basis = np.dot(X, R)

        basis2 = basis * basis.conjugate()
        basis3 = basis2 * basis
        W = np.diag(np.sum(basis2, axis=0))
        alpha = gamma / n_samples

        basis4 = np.dot(basis, W)
        transformed = np.dot(X.conjugate().T, (basis3 - (alpha * basis4)))
        U, svals, VT = np.linalg.svd(transformed)
        R = np.dot(U, VT)
        delta = np.sum(svals)
        if (abs(delta - delta_old) / delta) < rtol:
            converged = True
            break

    if(not converged):
        raise RuntimeError('Rotation process did not converge.')

    # De-normalize
    X = h[:, np.newaxis] * X

    # Rotate
    Xrot = np.dot(X, R)
    return Xrot, R


def pca(data:xr.DataArray, 
        dim:str, 
        normalize:bool=False,
        coslat_weights=None,
        detrend_degree:int=0,
        rotate:bool=False,
        n_modes_rot:int=10,
        )->xr.DataArray:
    """
    Performs EOF analysis by the use of numpys Singular Value Decomposition (SVD)

    Input:
    data:               Input data of type xr.DataArray
    dim:    string 
                        Dimension along which the correlation / covariance is to be calculated
                        (Usually 'time' (S-mode), but spatial correlations are possible as well)
    normalize:          boolean, default = False. If set to True, values are normalized by their standard deviation.
                        Corresponds to computing correlations instead of covariances (high absolute values have lower impact)
    coslat_weights:     Provide name of latitude coordinate as string in order to apply coslat weighting (accounts for smaller areas at higher latitudes).
                        e.g., if name of coslat_weights = 'lat', weights will be computed along dimension 'lat'
    detrend_degree:     Integer. Applies a polynomial fit of degree 'detrend_degree' along dim and subtracts this fit from the data.
                        Default is 0, which is equivalent to subtracting the mean (should always be done). detrend_degree=1 is a linear fit, 2 a quadratic fit etc..
    rotate:             Boolean, default is False. Performs Varimax rotation (Kaiser 1958, or Richman 1986) onto the first n_modes_rot which
                        were obtained from SVD.
                        Physical interpretation of standard EOF analysis might suffer under the mathematical property of orthogonality of the modes, 
                        which is often unphysical. Varimax rotation finds a new orthogonal basis, but without the property of uncorrelated PC's. 
                        Further (oblique) rotation methods are possible (Promax) but not implemented here.
    n_modes_rot:        integer, only used if option rotate was set to True see 'rotate' for explanation

    Returns
        pcs             Principal Components
        eof             Empirical Orthogonal functions
        exp_var_ratio   Fraction of explained variance by the respective mode
    """
    data = data.copy()

    # Remove the mean (and optionally detrend if detrend_degree>0)
    coeffs          = data.polyfit(dim=dim,deg=detrend_degree).polyfit_coefficients
    data_polyfit    = xr.polyval(data[dim],coeffs=coeffs)
    data            -= data_polyfit

    if normalize:
        # normalize by standard deviaton
        data /= data.std(dim)
        # since std can be zero valued, drop inf vals
        data = data.where(np.isfinite(data))
    
    if coslat_weights:
        latvals = data[coslat_weights].values
        weights = np.sqrt(np.cos(np.deg2rad(latvals)))
        weights = xr.DataArray(weights,coords={coslat_weights:latvals})
        data *= weights

    # combine the broadcast dimensions (e.g. reduce from 3D->2D):
    broadcast_dims = tuple([d for d in data.dims if d != dim])
    x = data.stack(dummy_dim=broadcast_dims)
    # drop every 'location' which contains nan values 
    x_clean = x.dropna(dim='dummy_dim')
    total_variance = (x_clean.std(dim,ddof=1)**2).sum().values

    # Any (M x N) matrix 'X' with (M > N) can be written as the product of three matrices 
    # X = P . L . Q^T
    # with cols of V being left singular vectors, cols of U being right singular vectors
    # and the diagonal matrix S containing the singular values

    # make use of numpy's singular value decomposition:
    scores, s, components = np.linalg.svd(x_clean, full_matrices=False)
    print(s.sum())

    if rotate:
        raise UserWarning('This option does not work yet. Please make use of a stable, already existing implementation (https://github.com/nicrie/xeofs)')

        #raise UserWarning('This method is not implemented yet')
        # only use a specific number for rotation:
        comp_slice     = components[:n_modes_rot,:].T # (has to be transposed for the varimax function)

        # the singular values explain the variance and must be applied to the eof
        s_scaled        = np.power(s[:n_modes_rot],2) / (data[dim].size - 1)
        comp_weighted   = comp_slice* np.sqrt(s_scaled)[np.newaxis,:]
        comp_rot,R      = varimax(comp_weighted)
        exp_var      = (abs(comp_rot)**2).sum(axis=0)
        exp_var_ratio = exp_var / total_variance
        comp_rot    = comp_rot / np.sqrt(exp_var)
        scores_slice = scores[:n_modes_rot].T
        scores_slice = np.dot(scores_slice,R)

        scores = scores_slice#.T
        components = comp_rot.T
        s = s[:n_modes_rot]
    else:
        # the fraction of explained variance for each mode is given by the singular values 
        exp_var_ratio = s**2 / (s**2).sum()
        n_modes_rot = data[dim].size
    # the option 'full_matrices=False' leads to a reduced size of the output matrices of size max(M,N)

    # force pcs to have unit variance (i.e. shift the variance to the eigen vectors):
    pc_std = scores.std(axis=0)
    pc      = scores / pc_std

    # store results into xarray dataarray

    mode = n_modes_rot - np.argsort(np.argsort(exp_var_ratio))
    pc = pc.T
    pc = xr.DataArray(
        pc,
        dims=('mode', dim),
        coords={'mode': mode, dim: x[dim]}
    )
    exp_var_ratio = xr.DataArray(exp_var_ratio,
                                 coords={'mode':mode})

    # rescale eofs in order to shift the units (variance) towards the EOFs 
    eof = components * pc_std[:, np.newaxis]
    eof = eof* s[:, np.newaxis]

    eofs = x.copy().rename({dim:'mode'}).isel(mode=slice(0,n_modes_rot))
    eofs[:, eofs[0].notnull().values] = eof

    # reinflate to original shape (e.g. 3D, incl. nan values)
    eofs = eofs.unstack(dim='dummy_dim')
    eofs.coords['mode'] = mode

    #sorting
    pc = pc.sortby('mode')
    eofs = eofs.sortby('mode')
    exp_var_ratio = exp_var_ratio.sortby('mode')
    
    
    return pc, eofs, exp_var_ratio
# %%
