"""Utilities."""

from collections.abc import Sequence
from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from healpix_geo.typing import ArrayLikeInt


def _check_depth(depth: ArrayLikeInt) -> None:
    """Check depth is in valid range [0, 29]."""
    depth_ = np.array(depth)
    if np.any(depth_ < 0) or np.any(depth_ > 29):
        raise ValueError("Depth must be in the [0, 29] closed range.")


def _check_ring(depth: int | NDArray[np.uint8], ring: int) -> None:
    """Check ring is in valid range [0, Nside]."""
    nside = 2 ** np.asarray(depth, dtype=np.uint64)

    if np.any(ring > nside):
        raise ValueError(
            "Crossing base cell boundaries more than once is not supported."
            f" Received ring={ring}, but expected an integer in the range of [0, {nside}]."
        )


def _check_cell_ids(
    cell_ids: NDArray[np.integer], depth: int | NDArray[np.uint8]
) -> None:
    """Check cell ids are in valid range [0, N].

    If depth is an array, check it has the same shape as cell ids.
    """
    depth_ = np.asarray(depth, dtype=np.uint64)
    npix = 12 * 4**depth_

    if np.any(cell_ids >= npix) or np.any(cell_ids < 0):
        raise ValueError(
            f"The input HEALPix cells contains a value out of [0, {npix - 1}]"
        )

    if depth_.ndim > 0 and cell_ids.shape != depth_.shape:
        raise ValueError(
            "A array valued depth must have the same shape as the cell ids"
        )


def validate_depth(depth: ArrayLikeInt) -> int | NDArray[np.uint8]:
    """Check depth value and return it as unsigned integer array."""
    depth_ = np.asarray(depth)
    _check_depth(depth_)
    return int(depth_.item()) if depth_.ndim == 0 else depth_.astype(np.uint8)


def validate_cell_ids(
    cell_ids: ArrayLikeInt, depth: int | ArrayLikeInt | None = None
) -> NDArray[np.uint64]:
    """Check cell ids values and return it as unsigned integer array.

    If depth is None, only check that the ids are positive, otherwise also check they
    are below max index and have same shape.
    """
    cell_ids_ = np.asarray(cell_ids)
    if depth is None:
        if np.any(cell_ids_ < 0):
            raise ValueError("The input HEALPix cells contains a negative value.")
    else:
        depth_ = np.asarray(depth)
        _check_cell_ids(cell_ids_, depth_)
    return np.atleast_1d(cell_ids_.astype(np.uint64))


_T = TypeVar("_T")


def validate_tuple2(seq: Sequence[_T], name: str) -> tuple[_T, _T]:
    """Check sequence is of length 2 and return as a tuple."""
    if len(seq) != 2:
        raise IndexError(f"Expected tuple of length 2 for argument '{name}'")
    return cast(tuple[_T, _T], tuple(seq))


def validate_tuple4(seq: Sequence[_T], name: str) -> tuple[_T, _T, _T, _T]:
    """Check sequence is of length 4 and return as a tuple."""
    if len(seq) != 4:
        raise IndexError(f"Expected tuple of length 4 for argument '{name}'")
    return cast(tuple[_T, _T, _T, _T], tuple(seq))
