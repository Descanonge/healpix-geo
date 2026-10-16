"""Utilities."""

from collections.abc import Sequence
from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from healpix_geo.typing import Array1D, ArrayLikeInt


def check_depth(depth: ArrayLikeInt) -> None:
    """Check depth is in valid range [0, 29]."""
    depth_ = np.array(depth)
    if np.any(depth_ < 0) or np.any(depth_ > 29):
        raise ValueError("Depth must be in the [0, 29] closed range.")


def check_ring(depth: int | NDArray[np.uint8], ring: int) -> None:
    """Check ring is in valid range [0, Nside]."""
    nside = 2 ** np.asarray(depth, dtype=np.uint64)

    if np.any(ring > nside):
        raise ValueError(
            "Crossing base cell boundaries more than once is not supported."
            f" Received ring={ring}, but expected an integer in the range of [0, {nside}]."
        )


def check_cell_ids(cell_ids: ArrayLikeInt, depth: ArrayLikeInt | None = None) -> None:
    """Check cell ids are in valid range [0, N].

    If depth is an array, check it has the same shape as cell ids.
    """
    cell_ids_ = np.atleast_1d(cell_ids)
    if depth is None:
        if np.any(cell_ids_ < 0):
            raise ValueError("The input HEALPix cells contains a negative value.")
        return

    depth_ = np.asarray(depth, dtype=np.uint64)
    npix = 12 * 4**depth_

    if depth_.ndim > 0 and cell_ids_.shape != depth_.shape:
        raise ValueError(
            "An array valued depth must have the same shape as the cell ids."
        )

    if np.any(cell_ids_ >= npix) or np.any(cell_ids_ < 0):
        raise ValueError(
            f"The input HEALPix cells contains a value out of [0, {npix - 1}]"
        )


def validate_depth(depth: ArrayLikeInt) -> int | NDArray[np.uint8]:
    """Check depth value and return it as unsigned integer array."""
    depth_ = np.asarray(depth)
    check_depth(depth_)
    return int(depth_.item()) if depth_.ndim == 0 else depth_.astype(np.uint8)


def validate_cell_ids(
    cell_ids: ArrayLikeInt, depth: int | ArrayLikeInt | None = None
) -> NDArray[np.uint64]:
    """Check cell ids values and return it as unsigned integer array.

    If depth is None, only check that the ids are positive, otherwise also check they
    are below max index and have same shape.
    """
    cell_ids_ = np.asarray(cell_ids)
    check_cell_ids(cell_ids_, depth)
    return np.atleast_1d(cell_ids_.astype(np.uint64))


_T = TypeVar("_T")


def _check_tuple_length(
    seq: Sequence | Array1D, length: int, *, name: str = ""
) -> None:
    """Check sequence is of correct length."""
    if len(seq) != length or (isinstance(seq, np.ndarray) and seq.ndim != 1):
        msg = f"Expected tuple of length {length}"
        if name:
            msg += f" for argument '{name}'"
        msg += f", received '{seq}'"
        raise IndexError(msg)


def validate_tuple2(seq: Sequence[float] | Array1D, name: str = "") -> tuple[_T, _T]:
    """Check sequence is of length 2 and return as a tuple."""
    _check_tuple_length(seq, 2, name=name)
    return cast(tuple[_T, _T], tuple(seq))


def validate_tuple4(
    seq: Sequence[_T] | Array1D, name: str = ""
) -> tuple[_T, _T, _T, _T]:
    """Check sequence is of length 4 and return as a tuple."""
    _check_tuple_length(seq, 4, name=name)
    return cast(tuple[_T, _T, _T, _T], tuple(seq))
