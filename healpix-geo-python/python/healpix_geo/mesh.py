from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from healpix_geo import _healpix_geo_python
from healpix_geo.utils import _check_depth, validate_cell_ids

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from healpix_geo.typing import ArrayLikeInt, EllipsoidLike


def vertex_to_lonlat(
    vertex_ids: ArrayLikeInt,
    depth: int,
    *,
    ellipsoid: EllipsoidLike = "sphere",
    num_threads: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Longitude and latitude coordinates for the given vertex ids

    Parameters
    ----------
    vertex_ids : array-like of int
        The given vertex ids.
    depth : int
        The depth of the cells the vertices are computed for.
    ellipsoid : ellipsoid-like, default: "sphere"
        Reference ellipsoid to evaluate the healpix vertex ids on.
    nthreads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    longitude, latitude : numpy.ndarray of numpy.float64
        The coordinates of the given vertices. Both arrays have the same shape as the array of vertex ids.

    Examples
    --------
    >>> from healpix_geo import vertex_to_lonlat
    >>> import numpy as np
    >>> depth = 0
    >>> vertex_ids = np.arange(12 * 4**depth + 2, dtype="uint64")
    >>> lon, lat = vertex_to_lonlat(vertex_ids, depth, ellipsoid="sphere")
    >>> np.stack([lon, lat], axis=-1)
    array([[  0.       ,  90.       ],
           [  0.       ,  41.8103149],
           [ 90.       ,  41.8103149],
           [180.       ,  41.8103149],
           [270.       ,  41.8103149],
           [ 45.       ,   0.       ],
           [135.       ,   0.       ],
           [225.       ,   0.       ],
           [315.       ,   0.       ],
           [  0.       , -41.8103149],
           [ 90.       , -41.8103149],
           [180.       , -41.8103149],
           [270.       , -41.8103149],
           [  0.       , -90.       ]])
    """
    _check_depth(depth)
    vertex_ids = validate_cell_ids(vertex_ids, depth)

    imax = 12 * 4**depth + 2
    if np.any(vertex_ids >= imax):
        raise ValueError(f"Some vertex ids are outside the valid range [0, {imax - 1}]")

    return _healpix_geo_python.vertex_to_lonlat(
        depth, vertex_ids, ellipsoid, num_threads
    )
