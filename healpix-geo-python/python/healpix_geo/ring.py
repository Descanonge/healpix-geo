from __future__ import annotations

from typing import TYPE_CHECKING, Any

import marray
import numpy as np

from healpix_geo import _healpix_geo_python
from healpix_geo.utils import (
    check_depth,
    check_ring,
    check_same_shape,
    validate_cell_ids,
    validate_depth,
    validate_tuple2,
    validate_tuple4,
)

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray

    from healpix_geo.typing import Array1D, ArrayLikeFloat, ArrayLikeInt, EllipsoidLike


def to_nested(
    ipix: ArrayLikeInt, depth: ArrayLikeInt, num_threads: int = 0
) -> NDArray[np.uint64]:
    """Convert from ring to nested

    Parameters
    ----------
    ipix : array-like of int
        The HEALPix cell indexes in the ring scheme.
    depth : int or array-like of int
        The HEALPix cell depth given as scalar or an array of the same shape as `ipix`.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    nested : numpy.ndarray of numpy.uint64
        The cell ids in the nested scheme.

    Examples
    --------
    >>> import healpix_geo
    >>> import numpy as np

    Scalar depth:

    >>> ipix_ring = np.array([1190, 189, 870, 432], dtype="uint64")
    >>> depth = 4
    >>> ipix_nested = healpix_geo.ring.to_nested(ipix_ring, depth)
    >>> ipix_nested
    array([ 32, 125,  45,  91], dtype=uint64)

    Array depth:

    >>> ipix_ring = np.array([44, 7, 9, 432], dtype="uint64")
    >>> depth = np.array([1, 3, 2, 4], dtype="uint8")
    >>> ipix_nested = healpix_geo.ring.to_nested(ipix_ring, depth)
    >>> ipix_nested
    array([ 32, 125,  45,  91], dtype=uint64)
    """
    depth_ = validate_depth(depth)
    ipix_ = validate_cell_ids(ipix, depth_)

    return _healpix_geo_python.ring.to_nested(ipix_, depth_, num_threads)


def to_zuniq(
    ipix: ArrayLikeInt, depth: ArrayLikeInt, num_threads: int = 0
) -> NDArray[np.uint64]:
    """Convert from nested to zuniq

    Parameters
    ----------
    ipix : array-like of int
        The HEALPix cell indexes in the ring scheme.
    depth : int or array-like of int
        The HEALPix cell depth given as scalar or an array of the same shape as `ipix`.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    zuniq : numpy.ndarray of numpy.uint64
        The cell ids in the zuniq scheme.

    Examples
    --------
    >>> import healpix_geo.nested
    >>> import numpy as np
    >>> ipix_ring = np.array([32, 125, 45, 91], dtype="uint64")
    >>> depth = np.array([1, 3, 2, 4], dtype="uint8")
    >>> ipix_zuniq = healpix_geo.ring.to_zuniq(ipix_ring, depth)
    >>> ipix_zuniq
    array([3530822107858468864,  806144333299318784,  882705526964617216,
           1106759608426299392], dtype=uint64)
    """
    depth_ = validate_depth(depth)
    ipix = validate_cell_ids(ipix, depth_)

    return _healpix_geo_python.ring.to_zuniq(ipix, depth_, num_threads)


def healpix_to_lonlat(
    ipix: ArrayLikeInt,
    depth: int,
    ellipsoid: EllipsoidLike = "sphere",
    num_threads: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Get the longitudes and latitudes of the center of some HEALPix cells.

    Parameters
    ----------
    ipix : array-like of int
        The HEALPix cell indexes.
    depth : `numpy.ndarray`
        The HEALPix cell depth.
    ellipsoid : str, default: "sphere"
        Reference ellipsoid to evaluate healpix on. If ``"sphere"``, this will return
        the same result as :py:func:`cdshealpix.ring.healpix_to_lonlat`.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    lon, lat : numpy.ndarray of numpy.float64
        The coordinates of the center of the HEALPix cells given as a longitude, latitude tuple.

    Raises
    ------
    ValueError
        When the HEALPix cell indexes given have values out of :math:`[0, 4^{29 - depth}[`.
    ValueError
        When the name of the ellipsoid is unknown.

    Examples
    --------
    >>> from healpix_geo.ring import healpix_to_lonlat
    >>> import numpy as np
    >>> ipix = np.array([42, 6, 10])
    >>> depth = 3
    >>> lon, lat = healpix_to_lonlat(ipix, depth, ellipsoid="WGS84")
    >>> lon
    array([ 45. , 112.5, 292.5])
    >>> lat
    array([60.54441647, 78.33504545, 78.33504545])
    """
    check_depth(depth)
    ipix = validate_cell_ids(ipix, depth)

    return _healpix_geo_python.ring.healpix_to_lonlat(
        depth, ipix, ellipsoid, num_threads
    )


def lonlat_to_healpix(
    longitude: ArrayLikeFloat,
    latitude: ArrayLikeFloat,
    depth: int,
    ellipsoid: EllipsoidLike = "sphere",
    num_threads: int = 0,
) -> NDArray[np.uint64]:
    r"""Get the HEALPix indexes that contains specific points.

    Parameters
    ----------
    lon : array-like of float
        The longitudes of the input points, in degrees.
    lat : array-like of float
        The latitudes of the input points, in degrees.
    depth : int
        The HEALPix cell depth.
    ellipsoid : str, default: "sphere"
        Reference ellipsoid to evaluate healpix on. If ``"sphere"``, this will return
        the same result as :py:func:`cdshealpix.ring.lonlat_to_healpix`.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    ipix : numpy.ndarray of numpy.uint64
        A numpy array containing all the HEALPix cell indexes.

    Raises
    ------
    ValueError
        When the number of longitudes and latitudes given do not match.
    ValueError
        When the name of the ellipsoid is unknown.

    Examples
    --------
    >>> from healpix_geo.ring import lonlat_to_healpix
    >>> import numpy as np
    >>> lon = np.array([0, 50, 25], dtype="float64")
    >>> lat = np.array([6, -12, 45], dtype="float64")
    >>> depth = 3
    >>> ipix = lonlat_to_healpix(lon, lat, depth, ellipsoid="WGS84")
    >>> ipix
    array([336, 436, 114], dtype=uint64)
    """
    check_depth(depth)
    longitude = np.atleast_1d(longitude).astype(np.float64)
    latitude = np.atleast_1d(latitude).astype(np.float64)

    check_same_shape(longitude=longitude, latitude=latitude)

    return _healpix_geo_python.ring.lonlat_to_healpix(
        depth, longitude, latitude, ellipsoid, num_threads
    )


def healpix_to_cartesian(
    ipix: ArrayLikeInt,
    depth: int,
    ellipsoid: EllipsoidLike = "sphere",
    num_threads: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Get the cartesian coordinates of the center of the given HEALPix cells.

    Parameters
    ----------
    ipix : array-like of int
        The HEALPix cell indexes.
    depth : int
        The HEALPix cell depth.
    ellipsoid : ellipsoid-like, default: "sphere"
        Reference ellipsoid to evaluate healpix on. If the reference ellipsoid
        is spherical, this will return the same result as
        :py:func:`cdshealpix.nested.healpix_to_lonlat`.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    x, y, z : numpy.ndarray of numpy.float64
        The coordinates of the center of the HEALPix cells.

    Raises
    ------
    ValueError
        When the HEALPix cell indexes given have values out of :math:`[0, 4^{29 - depth}[`.
    ValueError
        When the name of the ellipsoid is unknown.

    Examples
    --------
    >>> from healpix_geo.ring import healpix_to_cartesian
    >>> import numpy as np
    >>> ipix = np.array([42, 6, 10])
    >>> depth = 3
    >>> x, y, z = healpix_to_cartesian(ipix, depth, ellipsoid="WGS84")
    >>> x
    array([2223448.21176852, -495094.69891854,  495094.69891854])
    >>> y
    array([ 2223448.21176852,  1195264.33678817, -1195264.33678817])
    >>> z
    array([5530555.70358274, 6224606.75696243, 6224606.75696243])
    """
    check_depth(depth)
    ipix = validate_cell_ids(ipix, depth)

    return _healpix_geo_python.ring.healpix_to_cartesian(
        depth, ipix, ellipsoid, num_threads
    )


def cartesian_to_healpix(
    x: ArrayLikeFloat,
    y: ArrayLikeFloat,
    z: ArrayLikeFloat,
    depth: int,
    ellipsoid: EllipsoidLike = "sphere",
    num_threads: int = 0,
) -> NDArray[np.uint64]:
    r"""Get the HEALPix indexes that contain specific points.

    Parameters
    ----------
    x : array-like of float
        The x coordinate of the input points, in meters.
    y : array-like of float
        The y coordinate of the input points, in meters.
    z : array-like of float
        The z coordinate of the input points, in meters.
    depth : int
        The HEALPix cell depth.
    ellipsoid : ellipsoid-like, default: "sphere"
        Reference ellipsoid to evaluate healpix on. If the reference ellipsoid
        is spherical, this will return the same result as
        :py:func:`cdshealpix.nested.lonlat_to_healpix`.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    ipix : numpy.ndarray of numpy.uint64
        A numpy array containing all the HEALPix cell indexes stored as `np.uint64`.

    Raises
    ------
    ValueError
        When the number of longitudes and latitudes given do not match.
    ValueError
        When the name of the ellipsoid is unknown.

    Examples
    --------
    >>> from healpix_geo.ring import cartesian_to_healpix
    >>> import numpy as np
    >>> x = np.array(
    ...     [6343428.894701699, 4010777.6054728953, 4094327.79214653], dtype="float64"
    ... )
    >>> y = np.array([0.0, 4779858.620418726, 1909216.4044747741], dtype="float64")
    >>> z = np.array(
    ...     [662257.957592079, -1317402.5312295998, 4487348.408865919], dtype="float64"
    ... )
    >>> depth = 3
    >>> ipix = cartesian_to_healpix(x, y, z, depth, ellipsoid="WGS84")
    >>> ipix
    array([336, 436, 114], dtype=uint64)
    """
    check_depth(depth)
    x = np.atleast_1d(x).astype(np.float64)
    y = np.atleast_1d(y).astype(np.float64)
    z = np.atleast_1d(z).astype(np.float64)

    check_same_shape(x=x, y=y, z=z)

    return _healpix_geo_python.ring.cartesian_to_healpix(
        depth, x, y, z, ellipsoid, num_threads
    )


def vertices(
    ipix: ArrayLikeInt,
    depth: int,
    ellipsoid: EllipsoidLike,
    step: int = 1,
    num_threads: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get the longitudes and latitudes of the vertices of some HEALPix cells at a given depth.

    This method returns the 4 vertices of each cell in `ipix`.

    Parameters
    ----------
    ipix : array-like of int
        The HEALPix cell indexes.
    depth : int
        The depth of the HEALPix cells.
    ellipsoid : str, default: "sphere"
        Reference ellipsoid to evaluate healpix on. If ``"sphere"``, this will return
        the same result as :py:func:`cdshealpix.ring.vertices`.
    step : int, default: 1
        The number of vertices returned per HEALPix side. By default it is set to 1 meaning that
        it will only return the vertices of the cell. 2 means that it will return the vertices of
        the cell plus one more vertex per edge (the center of the edge). More generally, the number
        of vertices returned is ``4 * step``.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    longitude, latitude : numpy.ndarray of numpy.float64
        The sky coordinates of the 4 vertices of the HEALPix cells.
        `lon` and `lat` are of shape :math:`N` x :math:`4` arrays where N is the number of HEALPix cell given in `ipix`.

    Raises
    ------
    ValueError
        When the HEALPix cell indexes given have values out of :math:`[0, 4^{29 - depth}[`.

    Examples
    --------
    >>> from healpix_geo.ring import vertices
    >>> import numpy as np
    >>> ipix = np.array([42, 6, 10])
    >>> depth = 12
    >>> lon, lat = vertices(ipix, depth, ellipsoid="sphere")
    >>> np.stack([lon, lat], axis=-1)
    array([[[ 45.        ,  89.93147196],
            [ 54.        ,  89.9428933 ],
            [ 45.        ,  89.95431464],
            [ 36.        ,  89.9428933 ]],
    <BLANKLINE>
           [[120.        ,  89.96573598],
            [135.        ,  89.97715732],
            [ 90.        ,  89.98857866],
            [ 90.        ,  89.97715732]],
    <BLANKLINE>
           [[300.        ,  89.96573598],
            [315.        ,  89.97715732],
            [270.        ,  89.98857866],
            [270.        ,  89.97715732]]])
    """
    check_depth(depth)
    ipix = validate_cell_ids(ipix, depth)

    return _healpix_geo_python.ring.vertices(depth, ipix, ellipsoid, step, num_threads)


def vertex_indices(
    ipix: ArrayLikeInt,
    depth: int,
    *,
    num_threads: int = 0,
) -> NDArray[np.uint64]:
    """Get the indices of the 4 vertices of the given cells.

    Parameters
    ----------
    ipix : array-like of int
        The HEALPix cell indexes.
    depth : int
        The depth of the HEALPix cells.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    vertex_ids : numpy.ndarray of numpy.uint64
        The identifiers of vertices of the given cells.

    Raises
    ------
    ValueError
        When the HEALPix cell indexes given have values out of :math:`[0, 4^{29 - depth})`.

    See Also
    --------
    healpix_geo.vertex_to_lonlat

    Examples
    --------
    >>> from healpix_geo.nested import vertex_indices
    >>> import numpy as np

    >>> ipix = np.array([42, 6, 10])
    >>> depth = 12

    >>> vertex_ids = vertex_indices(ipix, depth)
    >>> vertex_ids
    array([[100542461, 100526078, 100509693, 100526077],
           [100608001, 100591618, 100575233, 100591617],
           [100607999, 100591616, 100575231, 100591615]], dtype=uint64)
    """
    check_depth(depth)
    ipix = validate_cell_ids(ipix, depth)

    return _healpix_geo_python.ring.vertex_indices(depth, ipix, num_threads)


def bilinear_interpolation(
    longitude: ArrayLikeFloat,
    latitude: ArrayLikeFloat,
    depth: int,
    *,
    ellipsoid: EllipsoidLike = "sphere",
    num_threads: int = 0,
) -> tuple[Any, Any]:
    """Get the cell ids and weights necessary to bilinearly interpolate the given values.

    The returned array are masked where the weights are zero. They are created using
    `marray <https://pypi.org/project/marray/>`__ and thus are compatible with the
    Python array API standard.

    Parameters
    ----------
    longitude : array-like of float
        The longitudes of the input points, in degrees.
    latitude : array-like of float
        The latitudes of the input points, in degrees.
    depth : int
        The depth of the HEALPix cells.
    ellipsoid : ellipsoid-like, default: "sphere"
        Reference ellipsoid to evaluate healpix on. If the reference ellipsoid
        is spherical, this will return the same result as
        :py:func:`cdshealpix.nested.vertices`.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    cell_ids : masked numpy.ndarray of numpy.uint64
        The neighbours above and below the given points as a :math:`N` x :math:`4` masked array.
    weights : masked numpy.ndarray of numpy.float64
        The associated weights as a :math:`N` x :math:`4` masked array.

    Raises
    ------
    ValueError
        When the HEALPix cell indexes given have values out of :math:`[0, 4^{29 - depth})`.

    Examples
    --------
    >>> from healpix_geo.ring import bilinear_interpolation
    >>> import numpy as np

    Define coordinates

    >>> lon = np.array([-15.0, -10.0, -5.0, 0.0, 5.0])
    >>> lat = np.array([30.0, 35.0, 40.0, 45.0, 50.0])

    Compute interpolation weights

    >>> cell_ids, weights = bilinear_interpolation(lon, lat, depth=6, ellipsoid="WGS84")
    >>> cell_ids
    MArray(
        array([[12661, 12405, 12404, 12149],
               [10872, 10617, 10616, 10360],
               [ 9340,  9085,  9084,  8828],
               [ 7320,  7080,  7563,  7319],
               [ 5943,  5727,  5726,  5514]], dtype=uint64),
        array([[False, False, False, False],
               [False, False, False, False],
               [False, False, False, False],
               [False, False, False, False],
               [False, False, False, False]])
    )
    >>> weights
    MArray(
        array([[0.22596183, 0.68795133, 0.02128467, 0.06480216],
               [0.15244623, 0.78751507, 0.00973729, 0.05030142],
               [0.04859157, 0.12318081, 0.23429192, 0.5939357 ],
               [0.32719215, 0.17280785, 0.32719215, 0.17280785],
               [0.14255714, 0.34522269, 0.1497    , 0.36252017]]),
        array([[False, False, False, False],
               [False, False, False, False],
               [False, False, False, False],
               [False, False, False, False],
               [False, False, False, False]])
    )
    """
    check_depth(depth)
    longitude = np.atleast_1d(longitude).astype(np.float64)
    latitude = np.atleast_1d(latitude).astype(np.float64)

    check_same_shape(longitude=longitude, latitude=latitude)

    ipix, weights = _healpix_geo_python.ring.bilinear_interpolation(
        depth, longitude, latitude, ellipsoid, num_threads
    )

    xp = marray.masked_namespace(np)
    mask = weights == 0

    return xp.asarray(ipix, mask=mask), xp.asarray(weights, mask=mask)


def neighbours(
    ipix: ArrayLikeInt,
    depth: int,
    *,
    connectivity: Literal["edge", "vertex", "all"] = "all",
    num_threads: int = 0,
) -> NDArray[np.int64]:
    """Get the direction-preserving immediate neighbours of some HEALPix cells at a given depth.

    Parameters
    ----------
    ipix : array-like of int
        The HEALPix cell indexes.
    depth : int
        The depth of the HEALPix cells.
    connectivity : {"edge", "vertex", "all"}, default: "all"
        The connectivity of the neighbours. The different kinds are:
        - ``"edge"`` connectivity returns the four edge-sharing neighbours in ``SW, NW, NE, SE`` order.
        - ``"vertex"`` connectivity returns the four vertex (but not edge) sharing neighbours. The order is ``S, W, N, E``.
        - ``"all"`` returns all immediate neighbours. The order is always ``S, SW, W, NW, N, NE, E, SE``.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    neighbours : numpy.ndarray of numpy.int64
        A :math:`N` x :math:`n` array, with :math:`n = 4` for edge or vertex
        connectivity and :math:`n = 8` for both together. Scalar inputs are
        normalized to one-dimensional input, producing shape ``(1, 4)`` or ``(1, 8)``.
        Missing directional positions are represented by ``-1`` and never shift
        other positions.
    """
    check_depth(depth)
    ipix = validate_cell_ids(ipix, depth)

    return _healpix_geo_python.ring.neighbours(depth, ipix, connectivity, num_threads)


def kth_neighbours(
    ipix: ArrayLikeInt, depth: int, ring: int, num_threads: int = 0
) -> NDArray[np.int64]:
    """Get the kth ring of neighbouring cells around some HEALPix cells at a given depth.

    This method returns a :math:`N` x :math:`8 k` `np.uint64` numpy array containing the neighbours of each cell of the :math:`N` sized `ipix` array.
    This method is wrapped around the `kth_neighbours <https://docs.rs/cdshealpix/0.9.1/cdshealpix/nested/struct.Layer.html#method.kth_neighbours>`__
    method of the `cdshealpix Rust crate <https://crates.io/crates/cdshealpix>`__.

    Parameters
    ----------
    ipix : array-like of int
        The HEALPix cell indexes.
    depth : int
        The depth of the HEALPix cells.
    ring : int
        The number of rings. `ring=0` returns just the input cell ids, `ring=1` returns the 8 (or 7) immediate
        neighbours, `ring=2` returns the 16 neighbours of the immediate neighbours, and so on.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    neighbours : numpy.ndarray of numpy.int64
        A :math:`N` x :math:`8 k` array containing the kth ring neighbours of each cell.

    Raises
    ------
    ValueError
        When the HEALPix cell indexes given have values out of :math:`[0, 4^{29 - depth}[`.

    Examples
    --------
    >>> from healpix_geo.ring import kth_neighbours
    >>> import numpy as np
    >>> ipix = np.array([42, 6, 10])
    >>> depth = 12
    >>> ring = 3
    >>> neighbours = kth_neighbours(ipix, depth, ring)
    >>> neighbours
    array([[ 83,  59,  39,  23,  11,   3,   2,  66,   1,   6,  15,  28,  45,
            225, 184, 147, 114,  85,  60,  65,  89, 117, 149, 185],
           [ 88,  63,  42,  25,  12,  44,  64,  38,  22,  10,  23,  51,  32,
              9,  19,  33, 123,  93,  67,  45,  31,  48,  69,  94],
           [102,  75,  52,  33,  18,  54,  76,  30,  16,   6,  17,  41,  24,
              5,  13,  25, 139, 107,  79,  55,  39,  58,  81, 108]])
    """
    check_depth(depth)
    check_ring(depth, ring)
    ipix = validate_cell_ids(ipix, depth)

    return _healpix_geo_python.ring.kth_neighbours(depth, ipix, ring, num_threads)


def kth_neighbourhood(
    ipix: ArrayLikeInt, depth: int, ring: int, num_threads: int = 0
) -> NDArray[np.int64]:
    """Get the kth ring neighbouring cells of some HEALPix cells at a given depth.

    This method returns a :math:`N` x :math:`(2 k + 1)^2` `np.uint64` numpy array containing the neighbours of each cell of the :math:`N` sized `ipix` array.
    This method is wrapped around the `kth_neighbourhood <https://docs.rs/cdshealpix/0.1.5/cdshealpix/nested/struct.Layer.html#method.neighbours_in_kth_ring>`__
    method from the `cdshealpix Rust crate <https://crates.io/crates/cdshealpix>`__.

    Parameters
    ----------
    ipix : array-like of int
        The HEALPix cell indexes.
    depth : int
        The depth of the HEALPix cells.
    ring : int
        The number of rings. `ring=0` returns just the input cell ids, `ring=1` returns the 8 (or 7) immediate
        neighbours, `ring=2` returns the 8 (or 7) immediate neighbours plus their immediate neighbours (a total of 24 cells), and so on.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    neighbours : numpy.ndarray of numpy.int64
        A :math:`N` x :math:`(2 k + 1)^2` array containing the kth ring neighbours of each cell.
        The :math:`5^{th}` element corresponds to the index of HEALPix cell from which the neighbours are evaluated.
        All its 8 neighbours occup the remaining elements of the line.

    Raises
    ------
    ValueError
        When the HEALPix cell indexes given have values out of :math:`[0, 4^{29 - depth}[`.

    Examples
    --------
    >>> from healpix_geo.ring import kth_neighbourhood
    >>> import numpy as np
    >>> ipix = np.array([42, 6, 10])
    >>> depth = 12
    >>> ring = 3
    >>> neighbours = kth_neighbourhood(ipix, depth, ring)
    >>> neighbours
    array([[ 42,  87,  62,  41,  25,  13,  26,  43,  63, 148, 115,  86,  61,
             40,  24,  12,   4,   0,   5,  14,  27,  44,  64,  88, 116,  83,
             59,  39,  23,  11,   3,   2,  66,   1,   6,  15,  28,  45, 225,
            184, 147, 114,  85,  60,  65,  89, 117, 149, 185],
           [  6,  14,   5,   0,  29,  15,   1,   7,  16,  43,  26,  13,   4,
             27,  11,   3,  18,   2,   8,  68,  46,  28,  17,  30,  47,  88,
             63,  42,  25,  12,  44,  64,  38,  22,  10,  23,  51,  32,   9,
             19,  33, 123,  93,  67,  45,  31,  48,  69,  94],
           [ 10,  20,   9,   2,  37,  21,   3,  11,  22,  53,  34,  19,   8,
             35,   7,   1,  12,   0,   4,  80,  56,  36,  23,  38,  57, 102,
             75,  52,  33,  18,  54,  76,  30,  16,   6,  17,  41,  24,   5,
             13,  25, 139, 107,  79,  55,  39,  58,  81, 108]])
    """
    check_depth(depth)
    check_ring(depth, ring)
    ipix = validate_cell_ids(ipix, depth)

    return _healpix_geo_python.ring.kth_neighbourhood(depth, ipix, ring, num_threads)


def angular_distances(
    from_: ArrayLikeInt, to_: ArrayLikeInt, depth: int, num_threads: int = 0
) -> NDArray[np.float64]:
    """Compute the angular distances

    Parameters
    ----------
    from_ : array-like of int
        The source Healpix cell indexes. Should be 1D.
    to_ : array-like of int
        The destination Healpix cell indexes. Should be 2D.
    depth : int
        The depth of the Healpix cells.
    num_threads : int, default: 0
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    distances : numpy.ndarray of numpy.float64
        The angular distances in radians.

    Raises
    ------
    ValueError
        When the Healpix cell indexes given have values out of :math:`[0, 4^{depth}[`.
    """
    check_depth(depth)

    from_ = validate_cell_ids(from_, depth)

    to_ = np.asarray(to_)
    mask = to_ != -1
    to_ = np.where(mask, to_, 0)
    to_ = validate_cell_ids(to_, depth)

    if from_.shape != to_.shape and from_.shape != to_.shape[:-1]:
        raise ValueError(
            "The shape of `from_` must be compatible with the shape of `to_`:\n"
            f"{to_.shape} or {to_.shape[:-1]} must be equal to {from_.shape}."
        )

    if from_.shape == to_.shape:
        intermediate_shape = to_.shape + (1,)
    else:
        intermediate_shape = to_.shape

    distances = _healpix_geo_python.ring.angular_distances(
        depth, from_, np.reshape(to_, intermediate_shape), num_threads
    )

    return np.where(mask, np.reshape(distances, to_.shape), np.nan)


def zone_coverage(
    bbox: tuple[float, float, float, float],
    depth: int,
    *,
    ellipsoid: EllipsoidLike = "sphere",
    flat: bool = True,
) -> tuple[Array1D[np.uint64], Array1D[np.uint8], Array1D[np.bool]]:
    """Search the cells covering the given bounding box

    Parameters
    ----------
    bbox : tuple of float
        The 2D bounding box to rasterize.
    depth : int
        The maximum depth of the cells to be returned.
    ellipsoid : ellipsoid-like, default: "sphere"
        Reference ellipsoid to evaluate healpix on.
    flat : bool, default: True
        If ``True``, the cells returned will all be at the passed depth.

    Returns
    -------
    cell_ids : numpy.ndarray
        The rasterized cell ids.
    depths : numpy.ndarray
        The depths of the cell ids. If ``flat is True``, these will all have the same
        value.
    fully_covered : numpy.ndarray
        Boolean array marking whether the cells are fully covered by the bounding box.
    """
    check_depth(depth)
    bbox = validate_tuple4(bbox, "bbox")

    return _healpix_geo_python.ring.zone_coverage(
        depth, bbox, ellipsoid=ellipsoid, flat=flat
    )


def box_coverage(
    center: tuple[float, float] | Array1D[np.floating],
    size: tuple[float, float] | Array1D[np.floating],
    angle: float,
    depth: int,
    *,
    ellipsoid: EllipsoidLike = "sphere",
    flat: bool = True,
) -> tuple[Array1D[np.uint64], Array1D[np.uint8], Array1D[np.bool]]:
    """Search the cells covering the given box.

    Parameters
    ----------
    center : tuple of float or numpy.ndarray of float
        The center of the box, either as a 2-sized array or as a 2-tuple of float.
    size : tuple of float or numpy.ndarray of float
        The size of the box, in degree, either as a 2-sized array of as a 2-tuple of
        float.
    angle : float
        The angle by which the box is rotated, in degree.
    depth : int
        The maximum depth of the cells to be returned.
    ellipsoid : ellipsoid-like, default: "sphere"
        Reference ellipsoid to evaluate healpix on.
    flat : bool, default: True
        If ``True``, the cells returned will all be at the passed depth.

    Returns
    -------
    cell_ids : numpy.ndarray of numpy.uint64
        The rasterized cell ids.
    depths : numpy.ndarray of numpy.uint8
        The depths of the cell ids. If ``flat is True``, these will all have the same
        value.
    fully_covered : numpy.ndarray of numpy.bool
        Boolean array marking whether the cells are fully covered by the box.
    """
    check_depth(depth)

    center = validate_tuple2(center, "center")
    size = validate_tuple2(size, "size")

    return _healpix_geo_python.ring.box_coverage(
        depth, center, size, angle, ellipsoid=ellipsoid, flat=flat
    )


def polygon_coverage(
    vertices: ArrayLikeFloat,
    depth: int,
    *,
    ellipsoid: EllipsoidLike = "sphere",
    flat: bool = True,
) -> tuple[Array1D[np.uint64], Array1D[np.uint8], Array1D[np.bool]]:
    """Search the cells covering the given polygon.

    Parameters
    ----------
    vertices : array-like of float
        The vertices of the polygon without holes. Must be of shape ``(n, 2)``.
    depth : int
        The maximum depth of the cells to be returned.
    ellipsoid : ellipsoid-like, default: "sphere"
        Reference ellipsoid to evaluate healpix on. If the reference ellipsoid is
        spherical, this will return the same result as
        :py:func:`cdshealpix.nested.polygon_search` followed by a translation to the ring
        scheme.
    flat : bool, default: True
        If ``True``, the cells returned will all be at the passed depth.

    Returns
    -------
    cell_ids : numpy.ndarray of numpy.uint64
        The rasterized cell ids.
    depths : numpy.ndarray of numpy.uint8
        The depths of the cell ids. If ``flat is True``, these will all have the same
        value.
    fully_covered : numpy.ndarray of numpy.bool
        Boolean array marking whether the cells are fully covered by the polygon.
    """
    check_depth(depth)

    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise IndexError("Vertices should be of shape (N, 2).")

    return _healpix_geo_python.ring.polygon_coverage(
        depth, vertices, ellipsoid=ellipsoid, flat=flat
    )


def cone_coverage(
    center: tuple[float, float] | Array1D[np.floating],
    radius: float,
    depth: int,
    *,
    delta_depth: int = 0,
    ellipsoid: EllipsoidLike = "sphere",
    flat: bool = True,
) -> tuple[Array1D[np.uint64], Array1D[np.uint8], Array1D[np.bool]]:
    """Search the cells covering the given cone

    Cone in this case means a circle on the surface of the reference ellipsoid.

    Parameters
    ----------
    center : tuple of float or numpy.ndarray of float
        The center of the cone, either as a 2-sized array or as a 2-tuple of float.
    radius : float
        The radius of the cone, in degree.
    depth : int
        The maximum depth of the cells to be returned.
    ellipsoid : ellipsoid-like, default: "sphere"
        Reference ellipsoid to evaluate healpix on. If the reference ellipsoid is
        spherical, this will return the same result as
        :py:func:`cdshealpix.nested.cone_search` followed by a translation to the ring
        scheme.
    flat : bool, default: True
        If ``True``, the cells returned will all be at the passed depth.

    Returns
    -------
    cell_ids : numpy.ndarray of numpy.uint64
        The rasterized cell ids.
    depths : numpy.ndarray of numpy.uint8
        The depths of the cell ids. If ``flat is True``, these will all have the same
        value.
    fully_covered : numpy.ndarray of numpy.bool
        Boolean array marking whether the cells are fully covered by the circle.
    """
    check_depth(depth)

    center = validate_tuple2(center, "center")

    return _healpix_geo_python.ring.cone_coverage(
        depth, center, radius, delta_depth=delta_depth, ellipsoid=ellipsoid, flat=flat
    )


def elliptical_cone_coverage(
    center: tuple[float, float] | Array1D[np.floating],
    ellipse_geometry: tuple[float, float] | Array1D[np.floating],
    position_angle: float,
    depth: int,
    *,
    delta_depth: int = 0,
    ellipsoid: EllipsoidLike = "sphere",
    flat: bool = True,
) -> tuple[Array1D[np.uint64], Array1D[np.uint8], Array1D[np.bool]]:
    """Search the cells covering the given elliptical cone.

    Elliptical cone in this case refers to an ellipse on the surface of the reference ellipsoid.

    Parameters
    ----------
    center : tuple of float or numpy.ndarray of float
        The center of the cone, either as a 2-sized array or as a 2-tuple of float.
    ellipse_geometry : tuple of float or numpy.ndarray of float
        The semimajor and semimajor axis, as a 2-sized array or as a 2-tuple of float.
    position_angle : float
        The orientation of the ellipse.
    depth : int
        The maximum depth of the cells to be returned.
    ellipsoid : ellipsoid-like, default: "sphere"
        Reference ellipsoid to evaluate healpix on. If the reference ellipsoid is
        spherical, this will return the same result as
        :py:func:`cdshealpix.nested.polygon_search` followed by a translation to the ring
        scheme.
    flat : bool, default: True
        If ``True``, the cells returned will all be at the passed depth.

    Returns
    -------
    cell_ids : numpy.ndarray of numpy.uint64
        The rasterized cell ids.
    depths : numpy.ndarray of numpy.uint8
        The depths of the cell ids. If ``flat is True``, these will all have the same value.
    fully_covered : numpy.ndarray of numpy.bool
        Boolean array marking whether the cells are fully covered by the ellipse.
    """
    check_depth(depth)

    center = validate_tuple2(center, "center")
    ellipse_geometry = validate_tuple2(ellipse_geometry, "ellipse_geometry")

    return _healpix_geo_python.ring.elliptical_cone_coverage(
        depth,
        center,
        ellipse_geometry,
        position_angle,
        delta_depth=delta_depth,
        ellipsoid=ellipsoid,
        flat=flat,
    )
